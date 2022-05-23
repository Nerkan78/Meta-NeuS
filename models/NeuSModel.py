from typing import List

from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, MultiSceneNeRF
from models.renderer import NeuSRenderer

import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pyhocon import ConfigFactory, ConfigTree
from pyhocon.converter import HOCONConverter

import random
import pathlib
import os
import time
import logging
import argparse
from shutil import copyfile
import contextlib


class Runner:
    def __init__(self, args, device):
        conf_path = args.neus_conf
        self.device = device# args.device

        # assert conf_path or checkpoint_path or args.extra_config_args, \
        #     "Specify at least config, checkpoint or extra_config_args"

        def update_config_tree(target: ConfigTree, source: ConfigTree, current_prefix: str = ''):
            """
            Recursively update values in `target` with those in `source`.

            current_prefix:
                str
                No effect, only used for logging.
            """
            for key in source.keys():
                if key not in target:
                    target[key] = source[key]
                else:
                    assert type(source[key]) == type(target[key]), \
                        f"Types differ in ConfigTrees: asked to update '{type(target[key])}' " \
                        f"with '{type(source[key])}' at key '{current_prefix}{key}'"

                    if type(source[key]) is ConfigTree:
                        update_config_tree(target[key], source[key], f'{current_prefix}{key}.')
                    else:
                        if target[key] != source[key]:

                            target[key] = source[key]

        # The eventual configuration, gradually filled from various sources
        # Config params resolution order: cmdline -> file -> checkpoint
        self.conf = ConfigFactory.parse_string("")
        if conf_path is not None:
            update_config_tree(self.conf, ConfigFactory.parse_file(conf_path))
        # if args.extra_config_args is not None:
        #     update_config_tree(self.conf, ConfigFactory.parse_string(args.extra_config_args))

        # Training parameters
        self.iter_step = 0
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        # List of (scene_idx, image_idx) pairs. Example: [[0, 4], [1, 2]].
        # -1 for random. Examples: [-1] or [[0, 4], -1]
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.base_learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_reduce_steps = \
            [int(x) for x in self.conf.get_list('train.learning_rate_reduce_steps')]
        self.learning_rate_reduce_factor = self.conf.get_float('train.learning_rate_reduce_factor')
        self.scenewise_layers_optimizer_extra_args = \
            dict(self.conf.get('train.scenewise_layers_optimizer_extra_args', default={}))
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.restart_from_iter = self.conf.get_int('train.restart_from_iter', default=None)

        self.use_fp16 = self.conf.get_bool('train.use_fp16', default=False)

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        if 'train.restart_from_iter' in self.conf:
            del self.conf['train']['restart_from_iter']

        self.finetune = self.conf.get_bool('train.finetune', default=False)
        parts_to_train = \
            self.conf.get_list('train.parts_to_train', default=[])
        load_optimizer = \
            self.conf.get_bool('train.load_optimizer', default=not self.finetune)
        parts_to_skip_loading = \
            self.conf.get_list('train.parts_to_skip_loading', default=[])
        # 'pick' or 'average'
        finetuning_init_algorithm = \
            self.conf.get_string('train.finetuning_init_algorithm', default='average')

        # For proper checkpoint auto-restarts
        for key in 'load_optimizer', 'restart_from_iter', 'parts_to_skip_loading':
            if f'train.{key}' in self.conf:
                del self.conf['train'][key]

        # if self.finetune:
        #     assert self.dataset.num_scenes == 1, "Can only finetune to one scene"
        #     assert self.dataset_val.num_scenes == 1, "Can only finetune to one scene"

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight', default=0.0)
        self.radiance_grad_weight = self.conf.get_float('train.radiance_grad_weight', default=0.0)

        # self.mode = args.mode
        self.model_list = []
        self.writer = None

        # Networks
        current_num_scenes = self.conf.get_int('dataset.original_num_scenes', default=1) #self.dataset.num_scenes)
        self.nerf_outside = MultiSceneNeRF(**self.conf['model.nerf'], n_scenes=current_num_scenes).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], n_scenes=current_num_scenes).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], n_scenes=current_num_scenes).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])



