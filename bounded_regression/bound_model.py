import torch
import numpy as np
import math
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import random
from operator import itemgetter
import matplotlib.pyplot as plt
import time
import yaml
from easydict import EasyDict


import trajnetplusplustools
from trajnetbaselines.lstm.lstm import drop_distant
from trajnetbaselines.lstm.run import draw_one_tensor
from trajnetbaselines.lstm.utils import seperate_xy, is_stationary, calc_fde_ade

#from diffusion
from  diffusion_bound_regression.MID_from_git.denoise_test import DataPreproc, DiffDenoiser

import diffusion_bound_regression.MID_from_git.models as models

from diffusion_bound_regression.MID_from_git.models.encoders.components.additive_attention import AdditiveAttention


class SmoothBounds():
    def __init__(self, 
                 model: torch.nn.Module,
                 device = torch.device('cpu'), 
                 sample_size:int=70,
                 time_noise_from_end:int=0,
                 pred_length = 12,
                 collision_treshold = 0.2,
                 obs_length = 9,
                 t_clean = 6, #part of obs we dont add noise on
                 t_noise = 3, #part of obs we add noise on
                 bound_margin = 0.2,
                 function = "mean"
                 ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.sample_size = sample_size
        self.time_noise_from_end = time_noise_from_end
        self.pred_length = pred_length
        self.collision_treshold = collision_treshold

        self._obs_length = obs_length
        self.bound_margin = bound_margin

        self.function = function
        if function == "diffusion":
            self._init_difusion(t_clean = t_clean, t_noise = t_noise)
    
    def _init_difusion(self, t_clean, t_noise, dt = 0.4, node_type = "PEDESTRIAN"):
        #only at begining
        model_path = "diffusion_bound_regression/MID_from_git/experiments/my_config_eval/eth_epoch60.pt"
        #model_path = "eth_epoch60.pt"
        config_path="diffusion_bound_regression/MID_from_git/configs/my_config_eval.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
            config = EasyDict(config)
        config["dataset"] = "eth"

        #print("HEEEEEEE")

        #instanciate data preprocessor
        self.data_prec = DataPreproc(node_type = node_type, dt = dt, t_clean = t_clean, t_noise = t_noise)
        #instanciate diffusion denoiser object
        self.dd = DiffDenoiser(config = config, model_path = model_path, dt = dt, node_type = node_type,
                               device=self.device)
        breakpoint()

    def preprocess_scenes(self, scenes: list, goals:list, remove_static:bool = False):
        """
        
        return
        ------
        :all_data: a list of tuple containing each scenes infos
        """
        #first preprocess the scenes
        all_data = []
        for i, (filename, scene_id, paths) in enumerate(scenes):

            scene = trajnetplusplustools.Reader.paths_to_xy(paths) # Now T_obs x N_agent x 2
            
            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            scene = torch.Tensor(scene).to(self.device)
            scene_goal = torch.Tensor(scene_goal).to(self.device)

            ## remove stationnary
            if remove_static:
                valid_scene = True
                for agent_path in paths:
                    xs, ys = seperate_xy(agent_path)
                    if is_stationary(xs, ys): #one or more is stationary
                        valid_scene = False #we skip this scnene
                #print(valid_scene)
                if not valid_scene:
                    continue

            all_data.append((scene_id, scene, scene_goal))
            #breakpoint()
            
        all_data = sorted(all_data, key=itemgetter(0))
        return all_data
    
    def compute_bounds_all(self,
                           all_data:list, 
                           filename:str, 
                           sigma:float, 
                           n0:int, 
                           r:float,
                           ) -> list:
        
        self.n0 = n0
        self.r = r
        self.sigma = sigma

        with open(filename, "w+") as f:
            f.write("scene_id\t" + "sigma\t" + "r\t" + "ade\t" + "fde\t" + "abd\t" + "fbd\n")

        start = 0
        all_real_pred = []
        all_mean_pred = []
        all_bounds = []
        for i in tqdm(range(start, min(len(all_data),self.sample_size))):
            x = all_data[i]
            scene_id = x[0]
            scene = x[1].to(self.device) #todevice is new
            scene_goal = x[2].to(self.device)
            batch_split = torch.Tensor([0,scene.size(1)]).to(self.device).long()

            observed = scene[:self._obs_length].detach().clone()

            #get real only with Tobs timesteps
            _, real_pred = self.model(observed.clone(),  
                                      scene_goal, 
                                      batch_split, 
                                      n_predict=self.pred_length)

            #breakpoint()
            #IMPORTANT : model precicts for all t!=0, so even for the one given (observation) -> replace them
            #HERE STICHING TO MAKE LEN == 21
            real_pred = torch.cat((observed, real_pred[-self.pred_length:]))
    
            #mean smoothing bounds computation
            mean_pred, bounds = self.compute_bounds_scene(observed, scene_goal, batch_split)

            #HERE STICHING TO MAKE LEN == 21
            mean_pred = torch.concat((observed, mean_pred[-self.pred_length:]))
            #breakpoint()

            #compute metrics : scene IS ground truth
            fde, ade = calc_fde_ade(mean_pred[-self.pred_length:], scene[-self.pred_length:]) 
            #compute final/average box dimensions
            fbd, abd = calc_fbd_abd(bounds)
            log(filename, scene_id, self.sigma, self.r, ade, fde, abd, fbd)

            #record
            all_real_pred.append(real_pred)
            all_mean_pred.append(mean_pred)
            all_bounds.append(bounds)

            


            
            

            #IMPORTANT : replacing Tobs with real obs is done inside func
            
            #all_pred.append(pred) 
            
            #breakpoint()

        return all_mean_pred, all_bounds, all_real_pred
    
    def compute_bounds_scene(self, observed, goal, batch_split):# -> tuple[torch.Tensor, torch.Tensor]:
        """
        computes the bounds according to corolary 2, eq9 in median smothing paper 
        note : this bound are based on the MEAN (not median)
        """
        if self.function == "mean":
            mean_pred, low_b, up_b = self.eval_g(observed, goal, batch_split, self.n0, diffusion=False)
        elif self.function == "diffusion":
            mean_pred, low_b, up_b = self.eval_g(observed, goal, batch_split, self.n0, diffusion=True)
        elif self.function == "median1":
            mean_pred, low_b, up_b = self.eval_g_median1(observed, goal, batch_split, self.n0)
        elif self.function == "median2":
            mean_pred, low_b, up_b = self.eval_g_median2(observed, goal, batch_split, self.n0)
        elif self.function == "compare":
            mean_pred, low_b, up_b = self.eval_g(observed, goal, batch_split, self.n0)
            med1_pred, low_b_m, up_b_m = self.eval_g_median1(observed, goal, batch_split, self.n0)
            med2_pred, low_b_m1, up_b_m2 = self.eval_g_median2(observed, goal, batch_split, self.n0)
            breakpoint()
        
        small_mp = mean_pred[-self.pred_length:]
        small_low_b = low_b[-self.pred_length:]
        small_up_b = up_b[-self.pred_length:]

        lb = small_low_b + (small_up_b-small_low_b)*norm.cdf( #cdf not ppf !
            (self.eval_eta(small_mp, small_low_b, small_up_b) - self.r)/self.sigma
        )

        #breakpoint( )

        ub = small_low_b + (small_up_b-small_low_b)*norm.cdf(
            (self.eval_eta(small_mp, small_low_b, small_up_b) + self.r)/self.sigma
        )

        bounds = torch.stack((lb, ub), dim=0)

        #breakpoint()

        return mean_pred, bounds

    def eval_g(self, observed, goal, batch_split,  num, diffusion = False):
        """
        Evaluate the function g(x) = E(f(x + eps)), where E is the expectation, f the model and eps~N(0,I*sig^2)
        of if diffusion = True
        Evaluate the function g(x) = E(f(denoise(x + eps))), where E is the expectation, f the model, 
        denoise the diffusion denoiser and eps~N(0,I*sig^2)
        """
        # smallest_pred = [None, None]
        # smallest_l2_norm = np.ones(self.num_classes) * np.inf
        
        with torch.no_grad():
            #necessary to put first iter here 
            if diffusion:
                outputs_perturbed, noise = self._sample_noise_diffusion(observed.detach().clone(), goal, batch_split)
            else:
                outputs_perturbed, noise = self._sample_noise(observed.detach().clone(), goal, batch_split)
            #IMPORTANT : model precicts for all t!=0, so even for the one given (observation) -> replace them
            #outputs_perturbed = torch.cat((observed + noise, outputs_perturbed[-self.pred_length:]))

            #breakpoint()

            mean_pred = outputs_perturbed.clone() #clone ! #keep nan is mean

            #remoove nans
            outputs_perturbed_neg10000 = outputs_perturbed.clone()
            pred_nan = torch.isnan(outputs_perturbed)
            for j in range(len(outputs_perturbed)):
                for k in range(len(outputs_perturbed[0])):
                    if any(pred_nan[j, k].tolist()):
                        outputs_perturbed.data[j, k] = 10000
                        outputs_perturbed_neg10000.data[j,k] = -10000
                        outputs_perturbed[j, k].detach()
                        outputs_perturbed_neg10000[j, k].detach()
            
            low_tot, _ = outputs_perturbed.view(-1, outputs_perturbed.shape[2]).min(axis=0)
            high_tot, _ = outputs_perturbed_neg10000.view(-1, outputs_perturbed_neg10000.shape[2]).max(axis=0)

            #lx, ly = torch.min(outputs_perturbed, )
            


            #visualize_scene(mean_pred)
            #breakpoint()

            for n in range(num - 1):
                #predict output
                outputs_perturbed, noise = self._sample_noise(observed.detach().clone(), goal, batch_split)

                #IMPORTANT : model precicts for all t!=0, so even for the one given (observation) -> replace them
                #outputs_perturbed = torch.cat((observed + noise, outputs_perturbed[-self.pred_length:]))

                mean_pred += outputs_perturbed #keep nan is mean

                outputs_perturbed_neg10000 = outputs_perturbed.clone()
                pred_nan = torch.isnan(outputs_perturbed)
                for j in range(len(outputs_perturbed)):
                    for k in range(len(outputs_perturbed[0])):
                        if any(pred_nan[j, k].tolist()):
                            outputs_perturbed.data[j, k] = 10000
                            outputs_perturbed_neg10000.data[j,k] = -10000
                            outputs_perturbed[j, k].detach()
                            outputs_perturbed_neg10000[j, k].detach()
                
                low, _ = outputs_perturbed.view(-1, outputs_perturbed.shape[2]).min(axis=0)
                high, _ = outputs_perturbed_neg10000.view(-1, outputs_perturbed_neg10000.shape[2]).max(axis=0)
                low_tot = torch.minimum(low_tot, low)
                high_tot = torch.maximum(high_tot, high)





                #visualize_scene(outputs_perturbed)
                #breakpoint()

            mean_pred /= num

            #add a margin
            diff = high_tot - low_tot
            low_tot -= diff*self.bound_margin
            high_tot += diff*self.bound_margin

            low_tot = low_tot.repeat(mean_pred.shape[0], mean_pred.shape[1], 1)
            high_tot = high_tot.repeat(mean_pred.shape[0], mean_pred.shape[1], 1)

            #breakpoint()

            return mean_pred, low_tot, high_tot

    def eval_g_median1(self, observed, goal, batch_split,  num):
        """
        returns the median1 instead of the mean, bounds are the same. 
        Median of type 1 is the real noisy traj that best represents the median
        """
        with torch.no_grad():
            noisy_preds = []
            low_tot = torch.Tensor([1000,1000])
            high_tot = torch.Tensor([-1000,-1000])
            for i in range(num):
                #compute noisy pred
                outputs_perturbed, _ = self._sample_noise(observed.detach().clone(), goal, batch_split)

                #append to the list of noisy predictions : keep nans in median
                noisy_preds.append(outputs_perturbed.clone())

                #replace nans by big value (and small to compute max corectly)
                outputs_perturbed_neg10000 = outputs_perturbed.clone()
                pred_nan = torch.isnan(outputs_perturbed)
                for j in range(len(outputs_perturbed)):
                    for k in range(len(outputs_perturbed[0])):
                        if any(pred_nan[j, k].tolist()):
                            outputs_perturbed.data[j, k] = 10000
                            outputs_perturbed_neg10000.data[j,k] = -10000
                            outputs_perturbed[j, k].detach()
                            outputs_perturbed_neg10000[j, k].detach()

                #compute lowest and highest x and y coord
                low, _ = outputs_perturbed.view(-1, outputs_perturbed.shape[2]).min(axis=0)
                high, _ = outputs_perturbed_neg10000.view(-1, outputs_perturbed_neg10000.shape[2]).max(axis=0)
                low_tot = torch.minimum(low_tot, low)
                high_tot = torch.maximum(high_tot, high)

            #convert list to tensor 
            noisy_preds = torch.stack(noisy_preds, dim = 0)
            
            #compute the traj that has the smallest distance to all the other -> the median traj
            smallest_dist = np.inf
            for i in range(num):
                dist = torch.sqrt(torch.nansum((torch.square(noisy_preds[:,-self.pred_length:,:,:]
                            - noisy_preds[i,-self.pred_length:,:,:].repeat( num, 1, 1, 1)))))
                if dist.item() < smallest_dist:
                    smallest_dist = dist.item()
                    idx = i

            #gather the median traj
            median = noisy_preds[idx]

            #add a margin to bounds
            diff = high_tot - low_tot
            low_tot -= diff*self.bound_margin
            high_tot += diff*self.bound_margin

            low_tot = low_tot.repeat(median.shape[0], median.shape[1], 1)
            high_tot = high_tot.repeat(median.shape[0], median.shape[1], 1)

            #breakpoint()

            return median, low_tot, high_tot


    def eval_g_median2(self, observed, goal, batch_split,  num):
        """
        returns the median2 instead of the mean, bounds are the same. 
        Median of type 2 is the artififial traj that has the midian of all coordinates.
        """
        with torch.no_grad():
            noisy_preds = []
            low_tot = torch.Tensor([1000,1000])
            high_tot = torch.Tensor([-1000,-1000])
            for i in range(num):
                #compute noisy pred
                outputs_perturbed, _ = self._sample_noise(observed.detach().clone(), goal, batch_split)

                #append to the list of noisy predictions : keep nans in median
                noisy_preds.append(outputs_perturbed.clone())

                #replace nans by big value (and small to compute max corectly)
                outputs_perturbed_neg10000 = outputs_perturbed.clone()
                pred_nan = torch.isnan(outputs_perturbed)
                for j in range(len(outputs_perturbed)):
                    for k in range(len(outputs_perturbed[0])):
                        if any(pred_nan[j, k].tolist()):
                            outputs_perturbed.data[j, k] = 10000
                            outputs_perturbed_neg10000.data[j,k] = -10000
                            outputs_perturbed[j, k].detach()
                            outputs_perturbed_neg10000[j, k].detach()

                #compute lowest and highest x and y coord
                low, _ = outputs_perturbed.view(-1, outputs_perturbed.shape[2]).min(axis=0)
                high, _ = outputs_perturbed_neg10000.view(-1, outputs_perturbed_neg10000.shape[2]).max(axis=0)
                low_tot = torch.minimum(low_tot, low)
                high_tot = torch.maximum(high_tot, high)

            #convert list to tensor 
            noisy_preds = torch.stack(noisy_preds, dim = 0) #100xTxNagentx2

            #gather the median traj
            median, _ = noisy_preds.nanmedian(dim=0) #squash the "sample" axis

            #add a margin to bounds
            diff = high_tot - low_tot
            low_tot -= diff*self.bound_margin
            high_tot += diff*self.bound_margin

            low_tot = low_tot.repeat(median.shape[0], median.shape[1], 1)
            high_tot = high_tot.repeat(median.shape[0], median.shape[1], 1)

            return median, low_tot, high_tot






    def eval_g_wrong_way(self, observed, goal, batch_split,  num):
        """
        WRONG : 
        Evaluate the function g(x) = E(f(x + eps)), where E is the expectation, f the model and eps~N(0,I*sig^2)
        """
        # smallest_pred = [None, None]
        # smallest_l2_norm = np.ones(self.num_classes) * np.inf

        with torch.no_grad():
            #necessary to put first iter here 
            outputs_perturbed, noise = self._sample_noise(observed.detach().clone(), goal, batch_split)
            #IMPORTANT : model precicts for all t!=0, so even for the one given (observation) -> replace them
            outputs_perturbed = torch.cat((observed + noise, outputs_perturbed[-self.pred_length:]))
            
            mean_pred = outputs_perturbed.clone() #clone !
            u = outputs_perturbed.clone() #clone !
            l = outputs_perturbed.clone() #clone !

            #visualize_scene(mean_pred)
            #breakpoint()

            for n in range(num - 1):
                #predict output
                outputs_perturbed, noise = self._sample_noise(observed.detach().clone(), goal, batch_split)

                #IMPORTANT : model precicts for all t!=0, so even for the one given (observation) -> replace them
                outputs_perturbed = torch.cat((observed + noise, outputs_perturbed[-self.pred_length:]))

                mean_pred += outputs_perturbed

                l = torch.minimum(l, outputs_perturbed)
                u = torch.maximum(u, outputs_perturbed)

                #visualize_scene(outputs_perturbed)
                #breakpoint()

            mean_pred /= num

            #breakpoint()

            return mean_pred, l, u
        
    def _sample_noise(self, observed: torch.tensor, goals: torch.tensor, batch_split):
        """
        produce a output from a noisy version on input
        """
        with torch.no_grad():
            if self.sigma != 0.0:
                noise = observed.detach().clone().normal_(mean = 0, std = self.sigma)
            else:
                noise = torch.zeros_like(observed)
            noise[:,1:,:] = 0 #modify only agent 0
            noise[:-self.time_noise_from_end,:,:] = 0 #add noise only on last timesteps
            # if no_noise_on_last:
            #     noise[-1,:,:] = 0 
            noisy_observation = observed + noise

            #run the model
            _, outputs_perturbed = self.model(
                noisy_observation, goals, batch_split, n_predict=self.pred_length
            )
            return outputs_perturbed, noise   
        
    def _sample_noise_diffusion(self, observed: torch.tensor, goals: torch.tensor, batch_split):
        """
        produce a output from a noisy version on input, but denoise it with diffusion
        """
        breakpoint()
        observation = observed
        batch, nodes, timesteps_o = self.data_prec.preproc_scene_only_obs(observation)

        observation_clean = self.data_prec.discard_nodes(observation, nodes) #discard ?? 

        breakpoint()


    def eval_eta(self, g, l, u):
        return self.sigma*norm.ppf((g - l)/(u - l))
    
    def eval_eta_with_resampling(self, observed, goal, batch_split,  num):
        """
        not used
        """
        mean_pred, l, u = self.eval_g(observed, goal, batch_split,  num)
        return self.sigma*norm.ppf((mean_pred - l)/(u - l))


#helper functions
def visualize_scene(scene, goal=None):
    for t in range(scene.shape[1]):
        path = scene[:, t]
        plt.plot(path[:, 0], path[:, 1])
    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])
    plt.show()


def calc_fbd_abd(bounds:torch.Tensor):
    lb, ub = bounds
    # if bounds.isnan().any():
    #     print("bounds is nan")
    #     breakpoint()

    box_dims = ub - lb
    #mean dimension of all box of all agents
    abd = box_dims.nanmean()
    #mean dimension of the FINAL box of all agents
    fbd = box_dims[-1].nanmean()

    #col = 0
    #iterate on time
    # for low_t, high_t in zip(lb, ub):
    #     if is_col
    #breakpoint()
    return fbd.item(), abd.item()


def log(filename, scene_id, sigma, r, ade, fde, abd, fbd):
    with open(filename,"a+") as f: 
        f.write(str(scene_id) + "\t" + str(sigma) + "\t" + str(r) + "\t" + 
                str(ade) + "\t" + str(fde) + "\t" + str(abd) + "\t" + str(fbd) + "\n")
