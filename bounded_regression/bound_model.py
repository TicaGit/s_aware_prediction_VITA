import torch
import numpy as np
import math
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import random
from operator import itemgetter
import matplotlib.pyplot as plt


import trajnetplusplustools
from trajnetbaselines.lstm.lstm import drop_distant
from trajnetbaselines.lstm.run import draw_one_tensor
from trajnetbaselines.lstm.utils import seperate_xy, is_stationary, calc_fde_ade


class SmoothBounds():
    def __init__(self, 
                 model: torch.nn.Module,
                 device = torch.device('cpu'), 
                 sample_size:int=70,
                 time_noise_from_end:int=0,
                 pred_length = 12,
                 collision_treshold = 0.2,
                 obs_length = 9,
                 ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.sample_size = sample_size
        self.time_noise_from_end = time_noise_from_end
        self.pred_length = pred_length
        self.collision_treshold = collision_treshold

        self._obs_length = obs_length

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
            f.write("scene_id\t" + "sigma\t" + "r\t" + "noise_norm\t" + "ade\t" + "fde\n")

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

            #IMPORTANT : model precicts for all t!=0, so even for the one given (observation) -> replace them
            real_pred = torch.cat((observed, real_pred[-self.pred_length:]))
            all_real_pred.append(real_pred)
            

            #mean smoothing bounds computation
            mean_pred, bounds = self.compute_bounds_scene(observed, scene_goal, batch_split)
            all_mean_pred.append(mean_pred)
            all_bounds.append(bounds)

            #fde, ade = calc_fde_ade(pred, scene) #scene IS ground truth
            

            #IMPORTANT : replacing Tobs with real obs is done inside func
            
            #all_pred.append(pred) 
            
            #breakpoint()

        return all_mean_pred, all_bounds, all_real_pred
    
    def compute_bounds_scene(self, observed, goal, batch_split) -> tuple[torch.Tensor, torch.Tensor]:
        """
        computes the bounds according to corolary 2, eq9 in median smothing paper 
        note : this bound are based on the MEAN (not median)
        """
        mean_pred, low_b, up_b = self.eval_g(observed, goal, batch_split, self.n0)
        
        small_mp = mean_pred[-self.pred_length:]
        small_low_b = low_b[-self.pred_length:]
        small_up_b = up_b[-self.pred_length:]

        lb = small_low_b + (small_up_b-small_low_b)*norm.cdf( #cdf not ppf !
            (self.eval_eta(small_mp, small_low_b, small_up_b) - self.r)/self.sigma
        )

        breakpoint( )

        ub = small_low_b + (small_up_b-small_low_b)*norm.cdf(
            (self.eval_eta(small_mp, small_low_b, small_up_b) + self.r)/self.sigma
        )

        bounds = torch.stack((lb, ub), dim=0)

        #breakpoint()

        return mean_pred, bounds



    def eval_g(self, observed, goal, batch_split,  num):
        """
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
