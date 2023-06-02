import torch
import pandas as pd
import numpy as np
import yaml
from easydict import EasyDict
import dill

import matplotlib.pyplot as plt
import time

from diffusion_bound_regression.MID_from_git.environment import Environment, Scene, Node, derivative_of

from diffusion_bound_regression.MID_from_git.models.encoders.dynamics.single_integrator import SingleIntegrator

#diffusion_bound_regression.MID_from_git
from diffusion_bound_regression.MID_from_git.utils.model_registrar import ModelRegistrar #but for torch.load() -> *** ModuleNotFoundError: No module named 'models'
#from diffusion_bound_regression.MID_from_git.utils.model_registrar import ModelRegistrar
from diffusion_bound_regression.MID_from_git.models.trajectron import Trajectron
from diffusion_bound_regression.MID_from_git.models.autoencoder import AutoEncoder
from diffusion_bound_regression.MID_from_git.dataset import EnvironmentDataset, collate, get_node_timestep_data#get_timesteps_data

#from diffusion_bound_regression.MID_from_git.models.encoders.components.additive_attention import AdditiveAttention

### other 
# import os
# import argparse
# import torch
# import dill
# import pdb
# import numpy as np
# import os.path as osp
# import logging
# import time
# from torch import nn, optim, utils
# import torch.nn as nn
# #from tensorboardX import SummaryWriter
# from tqdm.auto import tqdm
# import pickle

# from .dataset import EnvironmentDataset, collate, get_timesteps_data, restore
# from .models.autoencoder import AutoEncoder
# from .models.trajectron import Trajectron
# from .utils.model_registrar import ModelRegistrar
# from .utils.trajectron_hypers import get_traj_hypers
# import diffusion_bound_regression.MID_from_git.evaluation

import diffusion_bound_regression.MID_from_git.models as models
#copy from training
# import diffusion_bound_regression.MID_from_git.models
# import models.autoencoder
# import models.encoders
# import models.encoders.mgcvae
# import models.encoders.components
# import models.encoders.components.discrete_latent 
# import models.encoders.components.gmm2d
# import models.encoders.components.map_encoder
# import models.encoders.components.additive_attention 
# import models.encoders.components.n_pair_loss
# import models.encoders.model_utils
# import models.encoders.dynamics
# import models.encoders.dynamics.dynamic
# import models.encoders.dynamics.single_integrator
# print("models loaded")
# import sys
# print(sys.modules.keys())
# breakpoint()

# models
# models.autoencoder
# models.encoders
# models.encoders.mgcvae
# models.encoders.components
# models.encoders.components.discrete_latent 
# models.encoders.components.gmm2d
# models.encoders.components.map_encoder
# models.encoders.components.additive_attention 
# models.encoders.components.n_pair_loss
# models.encoders.model_utils
# models.encoders.dynamics
# models.encoders.dynamics.dynamic
# models.encoders.dynamics.single_integrator




#trajectron
import torch
import numpy as np
from diffusion_bound_regression.MID_from_git.models.encoders.mgcvae import MultimodalGenerativeCVAE
#from model.dataset import get_timesteps_data, restore
# import torch
# import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
container_abcs = collections.abc


#mcvae
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusion_bound_regression.MID_from_git.models.encoders.components import *
from diffusion_bound_regression.MID_from_git.models.encoders.model_utils import *
#import diffusion_bound_regression.MID_from_git.models.encoders.dynamics as dynamic_module #works
from diffusion_bound_regression.MID_from_git.models.encoders import dynamics as dynamic_module #looks better
from diffusion_bound_regression.MID_from_git.environment.scene_graph import DirectedEdge
from .utils import *
import pdb

#me
from diffusion_bound_regression.MID_from_git.models.encoders.components.additive_attention import *


# from models
from models import models
from models.encoders import encoders
import models.encoders.components
import models.encoders.components.additive_attention
from models.encoders.components.additive_attention import additive_attention

# from environment
import environment.environment 
import environment 
from environment.environment import Environment
from environment.scene import Scene
from environment.node import Node
from environment.node_type import NodeType, NodeTypeEnum
from environment.data_structures import DoubleHeaderNumpyArray



# import sys
# print(sys.modules.keys())
#breakpoint()

OBS_TENSOR = torch.Tensor([[[7.0400, 2.2700],
         [7.4500, 1.6900]],

        [[6.3600, 2.3400],
         [6.7600, 1.6500]],

        [[5.6500, 2.4500],
         [6.0700, 1.7200]],

        [[4.9300, 2.5900],
         [5.3700, 1.8700]],

        [[4.2300, 2.7600],
         [4.6600, 2.0700]],

        [[3.5400, 2.9700],
         [3.9400, 2.2700]],

        [[2.8800, 3.2000],
         [3.2100, 2.4500]],

        [[2.2500, 3.4400],
         [2.4900, 2.6200]],

        [[1.6300, 3.6700],
         [1.8000, 2.7800]]])

OBS_TENSOR_2 = torch.Tensor([[[ 7.3800, -1.5500],
         [    float('nan'),     float('nan')],
         [    float('nan'),     float('nan')]],

        [[ 6.9800, -1.5100],
         [ 7.2600, -1.1000],
         [    float('nan'),     float('nan')]],

        [[ 6.5400, -1.4800],
         [ 6.9100, -1.1100],
         [    float('nan'),     float('nan')]],

        [[ 6.0700, -1.4800],
         [ 6.5300, -1.0800],
         [    float('nan'),     float('nan')]],

        [[ 5.5800, -1.4800],
         [ 6.1200, -1.0100],
         [    float('nan'),     float('nan')]],

        [[ 5.0900, -1.4900],
         [ 5.6900, -0.9300],
         [    float('nan'),     float('nan')]],

        [[ 4.6000, -1.5100],
         [ 5.2400, -0.8600],
         [    float('nan'),     float('nan')]],

        [[ 4.1500, -1.5100],
         [ 4.7800, -0.8000],
         [-3.5300, -6.7400]],

        [[ 3.7300, -1.5000],
         [ 4.3000, -0.7600],
         [-3.1400, -5.9000]]])

OBS_TENSOR_3 = torch.Tensor([[[ 9.5700,  6.2400],
         [11.9400,  6.7700],
         [-1.8900,  4.3800], #fake
         [-1.7100,  5.1300], #fake
         [12.1600,  5.7500]],#fake

        [[ 9.0800,  6.2600],
         [11.4500,  6.9100],
         [-1.8900,  4.3800],
         [-1.7100,  5.1300],
         [12.1600,  5.7500]],

        [[ 8.5500,  6.3700],
         [10.8300,  6.8000],
         [-1.2800,  4.4600],
         [-1.1200,  5.1100],
         [11.6700,  5.8500]],

        [[ 8.1000,  6.4800],
         [10.3900,  6.7900],
         [-0.6900,  4.4400],
         [-0.5300,  5.0900],
         [11.1000,  5.9200]],

        [[ 7.6400,  6.5500],
         [ 9.8400,  6.8600],
         [-0.0800,  4.2400],
         [ 0.0900,  5.0200],
         [10.5100,  5.9100]],

        [[ 7.1700,  6.6200],
         [ 9.3600,  6.8500],
         [ 0.6000,  4.2300],
         [ 0.7600,  5.0000],
         [10.0000,  5.8900]],

        [[ 6.7300,  6.6400],
         [ 8.9500,  6.8000],
         [ 1.2000,  4.1700],
         [ 1.3600,  4.9300],
         [ 9.4400,  6.0000]],

        [[ 6.3400,  6.7100],
         [ 8.4100,  6.8700],
         [ 1.8000,  4.1100],
         [ 1.9600,  4.8700],
         [ 8.9100,  6.0300]],

        [[ 5.9900,  6.7700],
         [ 7.9500,  6.8600],
         [ 2.4500,  4.0900],
         [ 2.6000,  4.8500],
         [ 8.4100,  6.0200]]])

RAW_PRED_SLSTM = torch.Tensor([[[ 5.8722,  2.4143],
         [ 6.2698,  1.6188]],

        [[ 4.9980,  2.5349],
         [ 5.4165,  1.7782]],

        [[ 4.2281,  2.7107],
         [ 4.6747,  2.0157]],

        [[ 3.5503,  2.9122],
         [ 3.9837,  2.2635]],

        [[ 2.8863,  3.1750],
         [ 3.2749,  2.4484]],

        [[ 2.2521,  3.4191],
         [ 2.5389,  2.5840]],

        [[ 1.6505,  3.6554],
         [ 1.8144,  2.7530]],

        [[ 1.0350,  3.8694],
         [ 1.1391,  2.9163]],

        [[ 0.4285,  4.0292],
         [ 0.5048,  3.0762]],

        [[-0.1740,  4.1786],
         [-0.1114,  3.2248]],

        [[-0.7711,  4.3119],
         [-0.7303,  3.3671]],

        [[-1.3690,  4.4412],
         [-1.3412,  3.4983]],

        [[-1.9630,  4.5590],
         [-1.9505,  3.6274]],

        [[-2.5543,  4.6769],
         [-2.5535,  3.7468]],

        [[-3.1422,  4.7844],
         [-3.1531,  3.8649]],

        [[-3.7260,  4.8924],
         [-3.7472,  3.9741]],

        [[-4.3063,  4.9911],
         [-4.3371,  4.0817]],

        [[-4.8820,  5.0893],
         [-4.9221,  4.1812]],

        [[-5.4542,  5.1795],
         [-5.5027,  4.2783]]])

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

HYPERS = {   'batch_size': 256,
    'grad_clip': 1.0,
    'learning_rate_style': 'exp',
    'min_learning_rate': 1e-05,
    'learning_decay_rate': 0.9999,
    'prediction_horizon': 12,
    'minimum_history_length': 1,
    'maximum_history_length': 7,
    'map_encoder':
        {'PEDESTRIAN':
            {'heading_state_index': 6,
             'patch_size': [50, 10, 50, 90],
             'map_channels': 3,
             'hidden_channels': [10, 20, 10, 1],
             'output_size': 32,
             'masks': [5, 5, 5, 5],
             'strides': [1, 1, 1, 1],
             'dropout': 0.5
            }
        },
    'k': 1,
    'k_eval': 25,
    'kl_min': 0.07,
    'kl_weight': 100.0,
    'kl_weight_start': 0,
    'kl_decay_rate': 0.99995,
    'kl_crossover': 400,
    'kl_sigmoid_divisor': 4,
    'rnn_kwargs':
        {'dropout_keep_prob': 0.75},
    'MLP_dropout_keep_prob': 0.9,
    'enc_rnn_dim_edge': 128,
    'enc_rnn_dim_edge_influence': 128,
    'enc_rnn_dim_history': 128,
    'enc_rnn_dim_future': 128,
    'dec_rnn_dim': 128,
    'q_z_xy_MLP_dims': None,
    'p_z_x_MLP_dims': 32,
    'GMM_components': 1,
    'log_p_yt_xz_max': 6,
    'N': 1,
    'tau_init': 2.0,
    'tau_final': 0.05,
    'tau_decay_rate': 0.997,
    'use_z_logit_clipping': True,
    'z_logit_clip_start': 0.05,
    'z_logit_clip_final': 5.0,
    'z_logit_clip_crossover': 300,
    'z_logit_clip_divisor': 5,
    'dynamic':
        {'PEDESTRIAN':
            {'name': 'SingleIntegrator',
             'distribution': False,
             'limits': {}
            }
        },
    'state':
        {'PEDESTRIAN':
            {'position': ['x', 'y'],
             'velocity': ['x', 'y'],
             'acceleration': ['x', 'y']
            }
        },
    'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']}},
    'log_histograms': False,
    'dynamic_edges': 'yes',
    'edge_state_combine_method': 'sum',
    'edge_influence_combine_method': 'attention',
    'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
    'edge_removal_filter': [1.0, 0.0],
    'offline_scene_graph': 'yes',
    'incl_robot_node': False,
    'node_freq_mult_train': False,
    'node_freq_mult_eval': False,
    'scene_freq_mult_train': False,
    'scene_freq_mult_eval': False,
    'scene_freq_mult_viz': False,
    'edge_encoding': True,
    'use_map_encoding': False,
    'augment': True,
    'override_attention_radius': [],
    'learning_rate': 0.01,
    'npl_rate': 0.8,
    'K': 80,
    'tao': 0.4}

class DataPreproc():
    def __init__(self, node_type = "PEDESTRIAN", dt = 0.4, t_clean = 6, t_noise = 3, standardization = standardization):
        """
        scene : T x N_ag x 2 = 21 x Nag x 2
        """
        self.t_clean = t_clean
        self.t_noise = t_noise

        self.dt = dt                ## ?????

        self.node_type = node_type

        self.env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN)] = 3.0        ##???
        self.env.attention_radius = attention_radius

        self.data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    # def preproc_scene(self, scene_tensor:torch.Tensor):
    #     """
    #     params:
    #     -------

    #     scene : our format : T x N_ag x 2 = 21 x Nag x 2

    #     returns:
    #     --------

    #     batch : sea
    #     nodes : 
    #     timestep : 
    #     """
    #     obs_sc = scene_tensor.clone().numpy() #scene_tensor[:self.t_obs].numpy()

    #     t_tot = scene_tensor.shape[0]
    #     n_agents = scene_tensor.shape[1]

    #     scene = Scene(timesteps=self.t_obs+1, dt=self.dt, name="scene_custom", aug_func=None)

    #     for i_ag in range(n_agents):
    #         node = obs_sc[:,i_ag,:] #node == agent, now is Tx2

    #         breakpoint()
    #         new_first_idx = 0 #int(~node.isnan()).nonzero() #node_df['frame_id'].iloc[0] ##stil TODO ? all 0?, relative ? 

    #         x = node[:, 0]
    #         y = node[:, 1]
    #         vx = derivative_of(x, self.dt)
    #         vy = derivative_of(y, self.dt)
    #         ax = derivative_of(vx, self.dt)
    #         ay = derivative_of(vy, self.dt)

    #         data_dict = {('position', 'x'): x,
    #                      ('position', 'y'): y,
    #                      ('velocity', 'x'): vx,
    #                      ('velocity', 'y'): vy,
    #                      ('acceleration', 'x'): ax,
    #                      ('acceleration', 'y'): ay}
        
    #         node_data = pd.DataFrame(data_dict, columns=self.data_columns)
    #         node = Node(node_type=self.env.NodeType.PEDESTRIAN, node_id=str(i_ag), data=node_data)
    #         node.first_timestep = new_first_idx

    #         scene.nodes.append(node)

    #     self.env.scenes = [scene] #scenes is a list of 1 scene

    #     #breakpoint()
    #     timesteps = np.arange(0,self.t_obs)
    #     ht = 9-1 #no constrain on history or future
    #     ft = 12
    #     batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=self.node_type, state=HYPERS['state'],
    #             pred_state=HYPERS['pred_state'], edge_types=self.env.get_edge_types(),
    #             min_ht=ht, max_ht=ht, min_ft=ft, max_ft=ft, hyperparams=HYPERS)
    #     #breakpoint()
    #     return batch[0], batch[1], batch[2]


    def preproc_scene_only_obs(self, scene_tensor:torch.Tensor):
            """
            params:
            -------

            scene : our format : Tobs x N_ag x 2 = 9 x Nag x 2   #only obs

            returns:
            --------

            batch : sea
            nodes : 
            timestep : 
            """
            obs_sc = scene_tensor.clone().numpy() #scene_tensor[:self.t_obs].numpy()

            t_tot = scene_tensor.shape[0]
            n_agents = scene_tensor.shape[1]

            #scene should contain everything : all timesteps
            scene = Scene(timesteps=self.t_clean+self.t_noise, dt=self.dt, name="scene_custom", aug_func=None)

            for i_ag in range(n_agents):
                node = obs_sc[:,i_ag,:] #node == agent, now is Tx2
                

                new_first_idx = (np.isnan(node)).sum(axis=0)[0]

                x = node[:, 0]
                y = node[:, 1]
                x = x[~np.isnan(x)] #remoove nan : scene will begin at new_first_idx
                y = y[~np.isnan(y)]
                vx = derivative_of(x, self.dt)
                vy = derivative_of(y, self.dt)
                ax = derivative_of(vx, self.dt)
                ay = derivative_of(vy, self.dt)


                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}
            
                node_data = pd.DataFrame(data_dict, columns=self.data_columns)
                node = Node(node_type=self.env.NodeType.PEDESTRIAN, node_id=str(i_ag), data=node_data)
                node.first_timestep = new_first_idx

                scene.nodes.append(node)

            self.env.scenes = [scene] #scenes is a list of 1 scene
            #breakpoint()

            #breakpoint()
            timesteps = np.arange(0,self.t_clean)
            max_ht = self.t_clean-1 #6 first known
            min_ht = 1-1 #need to have a least 1 point ## 1-1 works ??'
            ft = self.t_noise     #we require that the 3 data are known
            #breakpoint() #voir comment nan sont geré ? node supr a cause de ht/ft ? 
            batch = self.get_timesteps_data_custom(env=self.env, scene=scene, t=timesteps, node_type=self.node_type, state=HYPERS['state'],
                    pred_state=HYPERS['pred_state'], edge_types=self.env.get_edge_types(),
                    min_ht=min_ht, max_ht=max_ht, min_ft=ft, max_ft=ft, hyperparams=HYPERS)
            #breakpoint()
            return batch[0], batch[1], batch[2]

    


    #OVERRIDDING A FUNCTION
    def get_timesteps_data_custom(self, env, scene, t, node_type, state, pred_state,
                        edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams):
        """
        Puts together the inputs for ALL nodes in a given scene and timestep in it.

        Overridd to not have duplicates of the same agent

        :param env: Environment
        :param scene: Scene
        :param t: Timestep in scene
        :param node_type: Node Type of nodes for which the data shall be pre-processed
        :param state: Specification of the node state
        :param pred_state: Specification of the prediction state
        :param edge_types: List of all Edge Types for which neighbors are pre-processed
        :param max_ht: Maximum history timesteps
        :param max_ft: Maximum future timesteps (prediction horizon)
        :param hyperparams: Model hyperparameters
        :return:
        """
        #breakpoint()
        nodes_per_ts = scene.present_nodes(t,
                                        type=node_type,
                                        min_history_timesteps=min_ht,
                                        min_future_timesteps=max_ft,
                                        return_robot=not hyperparams['incl_robot_node'])

        #only keep last timestep for each agent (avoid duplicates)

        new_nodes_per_ts = {} #empty dict
        used_agents = set() #set with wich agent have already been seen
        for k,v in sorted(nodes_per_ts.items(),  reverse = True): #go descending order
            ag = []
            for agent in v:
                if agent not in used_agents:
                    ag.append(agent)
                    used_agents.add(agent)
            if ag:
                new_nodes_per_ts[k] = ag
        
        #back in chrono order
        nodes_per_ts = dict(sorted(new_nodes_per_ts.items()))
        #breakpoint()
        batch = list()
        nodes = list()
        out_timesteps = list()
        #breakpoint()
        for timestep in nodes_per_ts.keys():
                scene_graph = scene.get_scene_graph(timestep,
                                                    env.attention_radius,
                                                    hyperparams['edge_addition_filter'],
                                                    hyperparams['edge_removal_filter'])
                present_nodes = nodes_per_ts[timestep]
                for node in present_nodes:
                    nodes.append(node)
                    out_timesteps.append(timestep)
                    batch.append(get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                                        edge_types, max_ht, max_ft, hyperparams,
                                                        scene_graph=scene_graph))
        if len(out_timesteps) == 0:
            return None
        return collate(batch), nodes, out_timesteps


    def discard_nodes(self, observation, nodes):
        present_nodes = set()
        for node in nodes:
            present_nodes.add(node.id)

        keep = []
        num_ag = observation.shape[1]
        keep = [str(i) in present_nodes for i in range(num_ag)] #bolean list
        return observation[:,keep,:], present_nodes


class DiffDenoiser():
    def __init__(self, config, model_path, dt=0.4, node_type = "PEDESTRIAN", device = torch.device("cuda"), beta_T = 0.05):
        self.device = device
        self.config = config

        self.dt = dt
        self.node_type = node_type

        #breakpoint()

        self.hyperparams = HYPERS
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # REGISTAR load for eval
        self.registrar = ModelRegistrar(model_path, self.device) #path correct ?? 

        print("olaaaa")

        #breakpoint()
        #import sys
        #sys.path.append('diffusion_bound_regression/MID_from_git/experiments/my_config_eval/')
        #sys.path.append('C:/Users/thiba/Desktop/Projet_VITA/code/my_s_aware_prediction_VITA/s_aware_prediction_VITA/diffusion_bound_regression/MID_from_git/models')

        #from pathlib import Path
        # str_path = "my_path"
        # path = Path(str_path)
        #model_path = "./experiments/my_config_eval/eth_epoch60.pt"
        #import os
        #model_path = os.path.join(self.model_dir, f"{self.config.dataset}_epoch{self.config.eval_at}.pt")
        from diffusion_bound_regression.MID_from_git.models.encoders.components.additive_attention import AdditiveAttention 
        checkpoint = torch.load(model_path, map_location = "cpu")
        # breakpoint()
        # ckpt_reg = torch.load("registar_dict.pt")
        # ckpt_reg = torch.load("ddpm_dict.pt")
        #breakpoint()

        self.registrar.load_models(checkpoint['encoder'])

        #ENCODER load for eval
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.train_data_path = "diffusion_bound_regression/MID_from_git/processed_data/eth_train.pkl"
        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()

        config = self.config
        model = AutoEncoder(config, encoder = self.encoder, beta_T=beta_T)

        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['ddpm']) #don't change to ddim (?)

        print("> Model built!")

    def get_context(self, batch):
        #breakpoint()
        context = self.model.encoder.get_latent(batch, self.node_type)
        return context

    def noise_with_diff(self, obs:torch.Tensor, sigma):
        """
        add a diffusion noise coresponding to sigma

        params:
        -------
        obs : tensor
        sigma : noise level to add

        returns:
        -------
        noisy_obs : 
        t_noise (int): the coresponding timestep (0-100)
        """
        #breakpoint()
        t_noise = self.get_t(sigma)
        alpha_t = self.model.diffusion.var_sched.alphas[t_noise].to("cpu")

        noisy_obs = obs*alpha_t.sqrt() + (1 - alpha_t).sqrt()*torch.randn_like(obs)

        return noisy_obs, t_noise

    def get_t(self, sigma):
        """
        return the timestep to the coresponding sigma, w.r.t a linear scheduler
        """
        beta1 = self.model.diffusion.var_sched.beta_1
        betaT = self.model.diffusion.var_sched.beta_T
        num_step = self.model.diffusion.var_sched.num_steps


        term = 1/(1+sigma**2)
        return round((1-beta1-term)/(betaT-beta1)*num_step)


    # def denoise_trough_pos_one_shot(self, t_noise, noisy_obs:torch.Tensor, context, Tpred = 12):
    #     """
    #     not finished !!!
    #     params:
    #     -------

    #     t_noise(int): timestep at which the noise (sigma) coresponds
    #     noisy_obs(torch.Tensor) :  must be in our format : Tpred x N_ag x 2 : IS THEPOSITION 
    #     context : encoding of the scene

    #     """

    #     var_sched = self.model.diffusion.var_sched #scheduler of Trajectron of autoencoder
    #     decoder_trans = self.model.diffusion.net #type TransformerConcatLinear

    #     batch_size = context.size(0)

    #     noisy_obs = noisy_obs.permute((1,0,2)) # now N_ag x Tpred x 2

    def denoise_trough_vel(self, t_noise, noisy_obs:torch.Tensor, context, stride=None, sampling="ddpm"):
        """
        params:
        -------

        t_noise(int): timestep at which the noise (sigma) coresponds
        noisy_obs(torch.Tensor) :  must be in our format : Tpred x N_ag x 2 : IS THEPOSITION 
        context : encoding of the scene

        """
        if stride == None:
            stride = t_noise #best : one shot
        elif stride > t_noise:
            raise Exception("invalid sride, must be smaller than noise number")

        var_sched = self.model.diffusion.var_sched #scheduler of Trajectron of autoencoder
        decoder_trans = self.model.diffusion.net #type TransformerConcatLinear

        batch_size = context.size(0)
        #breakpoint()

        noisy_obs = noisy_obs.permute((1,0,2)) # now N_ag x Tpred x 2

        v_t = self.get_velocity(noisy_obs).to(self.device) #x_T is veolity space
        #breakpoint()
        #verif

        flexibility = 0.0

        for t in range(t_noise, 0, -stride):

            z = torch.randn_like(v_t) if t > 1 else torch.zeros_like(v_t)
            alpha = var_sched.alphas[t]
            alpha_bar = var_sched.alpha_bars[t]
            alpha_bar_next = var_sched.alpha_bars[t-stride]

            sigma = var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)


            beta = var_sched.betas[[t]*batch_size]
            #breakpoint()
            e_theta = decoder_trans(v_t, beta=beta, context=context)
            if sampling == "ddpm":
                v_next = c0 * (v_t - c1 * e_theta) + sigma * z
            elif sampling == "ddim":
                v0_t = (v_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                v_next = alpha_bar_next.sqrt() * v0_t + (1 - alpha_bar_next).sqrt() * e_theta

            v_t = v_next    # Stop gradient and save trajectory.




            # beta = var_sched.betas[[t]*batch_size]
            # e_theta = decoder_trans(v_t, beta=beta, context=context)

            # v0_t = (v_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            # v_next = alpha_bar_next.sqrt() * v0_t + (1 - alpha_bar_next).sqrt() * e_theta

            # v_t = v_next.detach()     # Stop gradient and save trajectory.

        dynamics = self.model.encoder.node_models_dict[self.node_type].dynamic
        obs_denoised = dynamics.integrate_samples(v_t.unsqueeze(0)).squeeze(0) # Dynamics is expecting a batch of speed !!!!

        obs_denoised = obs_denoised.permute((1,0,2)) # now Tpred x N_ag x 2
        #breakpoint()

        #other
        #pos = self.integrate_velocity(v_t, last pos of unnoised obs (6th value, pos 5))


        return obs_denoised
    
    def integrate_velocity(self, init_pos:torch.tensor, vel:torch.Tensor):
        return torch.cumsum(vel, dim=1) * self.dt + init_pos                ##carefull wich dim



    def get_velocity(self, pos:torch.Tensor):
        """
        pos() : N_ag x Tpred x 2
        """
        pos = pos.numpy()
        num_agents = pos.shape[0]
        vel = []
        for i_ag in range(num_agents):
            x = pos[i_ag, :, 0]
            y = pos[i_ag, :, 1]
            vx = torch.from_numpy(derivative_of(x, self.dt))
            vy = torch.from_numpy(derivative_of(y, self.dt))
            vel.append(torch.stack((vx,vy), dim = 1))
        return torch.stack(vel, dim = 0)

     

# def eval_senes(scenes:list(Scene)):
#     for i, scene in enumerate(scenes):
#             print(f"----- Evaluating Scene {i + 1}/{len(scenes)}")

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_type = "PEDESTRIAN"
    dt = 0.4                        ##???
    #t_pred = 12
    #t_obs = 9
    t_noise = 3
    t_clean = 6
    beta_T = 0.05          #orig, trained with 0.05          ##final variance ? 

    data_prec = DataPreproc(node_type = node_type, dt = dt, t_clean = t_clean, t_noise = t_noise)
    #scene_test = torch.rand((21,4,2))
    #scene_test = torch.cat((OBS_TENSOR[:t_obs,:,:], RAW_PRED_SLSTM[-t_pred:,:,]), dim = 0)
    #breakpoint()

    time_before = time.perf_counter()

    observation = OBS_TENSOR_3

    #batch, nodes, timesteps_o = data_prec.preproc_scene(scene_test) #whole scene
    batch, nodes, timesteps_o = data_prec.preproc_scene_only_obs(observation) #konw obs is futur

    #discard agent with no "obs" in the first 6 -> can't be denoised
    #breakpoint()
    observation_clean = data_prec.discard_nodes(observation, nodes)

    #only at begining
    model_path = "experiments/my_config_eval/eth_epoch60.pt"
    config_path="configs/my_config_eval.yaml"
    with open(config_path) as f:
       config = yaml.safe_load(f)
    config = EasyDict(config)
    dd = DiffDenoiser(config = config, model_path = model_path, dt = dt, node_type = node_type, device=device, beta_T=beta_T)

    ##
    context = dd.get_context(batch) #Nag x 256

    #WE NOISE ON OBS, NOT PRED
    #pred = scene_test[-t_pred:,:,:].clone() #now 12xnagx2
    #obs = scene_test[:t_obs,:,:].clone() 
    last_3_obs = observation_clean[-t_noise:,:,:].clone().detach()

    sigma = 0.1
    #breakpoint()
    noisy_last_3_obs, t_coresp_noise = dd.noise_with_diff(last_3_obs, sigma)
    #noisy_obs = obs
    #t_noise = 47

    #do all in one step
    denoised_last_3_obs = dd.denoise_trough_vel(t_coresp_noise, noisy_last_3_obs, context, sampling="ddim")
    denoised_last_3_obs = denoised_last_3_obs.cpu().detach()

    time_after = time.perf_counter()
    print("time : ", time_after - time_before)

    #same, but step by step
    denoised_last_3_obs_stride_1 = dd.denoise_trough_vel(t_noise, noisy_last_3_obs, context, stride = 1, sampling="ddim")
    denoised_last_3_obs_stride_1 = denoised_last_3_obs_stride_1.cpu().detach()
    #breakpoint()

    nag = denoised_last_3_obs.shape[1]
    for i, i_ag in enumerate(range(nag)):
        #obs_i = observation_clean[:,i_ag,:]
        obs_i = observation_clean[:(t_clean + 1),i_ag,:]
        last_3_obs_i = last_3_obs[:,i_ag,:]
        noisy_3_obs_i = noisy_last_3_obs[:,i_ag,:]
        denoised_3_obs_i = denoised_last_3_obs[:,i_ag,:]
        denoised_3_obs_i_s1 = denoised_last_3_obs_stride_1[:,i_ag,:]

        if i == 0:
            #plt.plot(obs_i[:,0], obs_i[:,1], c="r", label="real obs")
            plt.plot(obs_i[:,0], obs_i[:,1], c="r", label="untouched")
            plt.plot(last_3_obs_i[:,0], last_3_obs_i[:,1], c="y", label="will be noised")
            plt.plot(noisy_3_obs_i[:,0], noisy_3_obs_i[:,1], c="g", label="noised")
            plt.plot(denoised_3_obs_i[:,0], denoised_3_obs_i[:,1], c="cyan", label="denoise (only one step)")
            plt.plot(denoised_3_obs_i_s1[:,0], denoised_3_obs_i_s1[:,1], c="b", label="many steps (1 by 1)")
        else:
            #plt.plot(obs_i[:,0], obs_i[:,1], c="r")
            plt.plot(obs_i[:,0], obs_i[:,1], c="r")
            plt.plot(last_3_obs_i[:,0], last_3_obs_i[:,1], c="y")
            plt.plot(noisy_3_obs_i[:,0], noisy_3_obs_i[:,1], c="g")
            plt.plot(denoised_3_obs_i[:,0], denoised_3_obs_i[:,1], c="cyan")
            plt.plot(denoised_3_obs_i_s1[:,0], denoised_3_obs_i_s1[:,1], c="b")
    
    plt.legend()
    plt.savefig('plot_diffusion_denoise.png')
    #time.sleep(5)
    #YESSS !
    
    
    #eval_senes()
    #breakpoint()
    scenes = []



    traj = 3

if __name__ == "__main__":
    main()