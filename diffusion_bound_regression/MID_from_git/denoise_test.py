import torch
import pandas as pd
import numpy as np
import yaml
from easydict import EasyDict
import dill

from environment import Environment, Scene, Node, derivative_of

from utils.model_registrar import ModelRegistrar
from models.trajectron import Trajectron
from models.autoencoder import AutoEncoder
from dataset import EnvironmentDataset, collate, get_timesteps_data

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
    def __init__(self, node_type = "PEDESTRIAN", dt = 0.4, t_obs = 9, t_pred = 12, standardization = standardization):
        """
        scene : T x N_ag x 2 = 21 x Nag x 2
        """
        self.t_obs = t_obs
        self.t_pred = t_pred

        self.dt = dt                ## ?????

        self.node_type = node_type

        self.env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN)] = 3.0        ##???
        self.env.attention_radius = attention_radius

        self.data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    def preproc_scene(self, scene_tensor:torch.Tensor):
        """
        params:
        -------

        scene : our format : T x N_ag x 2 = 21 x Nag x 2

        returns:
        --------

        batch : sea
        nodes : 
        timestep : 
        """
        obs_sc = scene_tensor.clone().numpy() #scene_tensor[:self.t_obs].numpy()

        t_tot = scene_tensor.shape[0]
        n_agents = scene_tensor.shape[1]

        scene = Scene(timesteps=self.t_obs+1, dt=self.dt, name="scene_custom", aug_func=None)

        for i_ag in range(n_agents):
            node = obs_sc[:,i_ag,:] #node == agent, now is Tx2

            new_first_idx = 0 #int(~node.isnan()).nonzero() #node_df['frame_id'].iloc[0] ##stil TODO ? all 0?, relative ? 

            x = node[:, 0]
            y = node[:, 1]
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
        timesteps = np.arange(0,self.t_obs)
        ht = 9-1 #no constrain on history or future
        ft = 12
        batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=self.node_type, state=HYPERS['state'],
                pred_state=HYPERS['pred_state'], edge_types=self.env.get_edge_types(),
                min_ht=ht, max_ht=ht, min_ft=ft, max_ft=ft, hyperparams=HYPERS)
        #breakpoint()
        return batch[0], batch[1], batch[2]


# def get_scenes(data:list(torch.Tensor)) -> list(Scene):
#     """
#     return format is then used like in orig file.
#     One scene per dataset + train/test : e.g. eth_train is ONE scene
#     """
#     env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)

#     attention_radius = dict()
#     attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
#     env.attention_radius = attention_radius

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

        checkpoint = torch.load(model_path, map_location = "cpu")
        self.registrar.load_models(checkpoint['encoder'])

        #ENCODER load for eval
        self.encoder = Trajectron(self.registrar, self.hyperparams, self.device)

        self.train_data_path = "processed_data/eth_train.pkl"
        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        self.encoder.set_environment(self.train_env) #FDP !!!
        self.encoder.set_annealing_params()

        config = self.config
        model = AutoEncoder(config, encoder = self.encoder, beta_T=beta_T)

        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['ddpm']) #don't change to ddim (?)

        print("> Model built!")

    def get_context(self, batch):
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
        t_noise = self.get_t(sigma)
        breakpoint()
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


    def denoise_trough_pos_one_shot(self, t_noise, noisy_obs:torch.Tensor, context, Tpred = 12):
        """
        not finished !!!
        params:
        -------

        t_noise(int): timestep at which the noise (sigma) coresponds
        noisy_obs(torch.Tensor) :  must be in our format : Tpred x N_ag x 2 : IS THEPOSITION 
        context : encoding of the scene

        """

        var_sched = self.model.diffusion.var_sched #scheduler of Trajectron of autoencoder
        decoder_trans = self.model.diffusion.net #type TransformerConcatLinear

        batch_size = context.size(0)

        noisy_obs = noisy_obs.permute((1,0,2)) # now N_ag x Tpred x 2

    def denoise_trough_vel(self, t_noise, noisy_obs:torch.Tensor, context, Tobs = 9, stride=None):
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

        noisy_obs = noisy_obs.permute((1,0,2)) # now N_ag x Tpred x 2

        v_t = self.get_velocity(noisy_obs).to(self.device) #x_T is veolity space
        breakpoint()
        #verif

        for t in range(t_noise, 0, -stride):
            #z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            #alpha = var_sched.alphas[t]
            alpha_bar = var_sched.alpha_bars[t]
            alpha_bar_next = var_sched.alpha_bars[t-stride]
            #pdb.set_trace()

            #only for ddpm ( not us)
            # sigma = var_sched.get_sigmas(t, flexibility)
            # c0 = 1.0 / torch.sqrt(alpha)
            # c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            # if sampling == "ddpm":
            #     x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            # elif sampling == "ddim":



            beta = var_sched.betas[[t]*batch_size]
            e_theta = decoder_trans(v_t, beta=beta, context=context)

            v0_t = (v_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            v_next = alpha_bar_next.sqrt() * v0_t + (1 - alpha_bar_next).sqrt() * e_theta

            v_t = v_next.detach()     # Stop gradient and save trajectory.

        breakpoint()

        dynamics = self.model.encoder.node_models_dict[self.node_type].dynamic
        obs_denoised = dynamics.integrate_samples(v_t)

        obs_denoised = obs_denoised.permute((1,0,2)) # now Tpred x N_ag x 2

        return obs_denoised


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

    node_type = "PEDESTRIAN"
    dt = 0.4                        ##???
    t_pred = 12
    t_obs = 9
    beta_T = 0.05          #orig, trained with            ##final variance ? 

    data_prec = DataPreproc(node_type = node_type, dt = dt, t_pred = t_pred)
    #scene_test = torch.rand((21,4,2))
    scene_test = torch.cat((OBS_TENSOR[:t_obs,:,:], RAW_PRED_SLSTM[-t_pred:,:,]), dim = 0)
    breakpoint()

    batch, nodes, timesteps_o = data_prec.preproc_scene(scene_test)

    #only at begining
    model_path = "experiments/my_config_eval/eth_epoch60.pt"
    config_path="configs/my_config_eval.yaml"
    with open(config_path) as f:
       config = yaml.safe_load(f)
    config = EasyDict(config)
    dd = DiffDenoiser(config = config, model_path = model_path, dt = dt, node_type = node_type, beta_T=beta_T)

    ##
    context = dd.get_context(batch) #Nag x 256

    #WE NOISE ON OBS, NOT PRED
    #pred = scene_test[-t_pred:,:,:].clone() #now 12xnagx2
    obs = scene_test[:t_obs,:,:].clone() 

    sigma = 0.1
    breakpoint()
    noisy_obs, t_noise = dd.noise_with_diff(obs, sigma)
    #noisy_obs = obs
    #t_noise = 47
    denoised_pred = dd.denoise_trough_vel(t_noise, noisy_obs, context)
    breakpoint()
    pass
    
    
    
    #eval_senes()

    scenes = []



    traj = 3

if __name__ == "__main__":
    main()