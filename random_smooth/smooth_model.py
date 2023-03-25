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



class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, 
                 slstm: torch.nn.Module,
                 device = torch.device('cpu'), 
                 sample_size:int=70,
                 time_noise_from_end=0,
                 pred_length = 12,
                 collision_treshold = 0.2,
                 ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.slstm = slstm
        self.device = device
        self.sample_size = sample_size
        self.time_noise_from_end = time_noise_from_end
        self.pred_length = pred_length
        self.collision_treshold = collision_treshold

        self.num_classes = 2 #binary, col or no_col


    def certify_all(self, scenes: list, goals:list, filename_results:str, 
                    sigma:int, n0: int, n: int, 
                    alpha: float, n_predict:int =12):
        """
        cerfify all scenes
        """
        self.n0 = n0
        self.n = n
        self.alpha = alpha
        self.n_predict = n_predict
        self.sigma = sigma 

        #random.shuffle(scenes)
        check_point_size = 50
        all_data = []


        #first preprocess the scenes
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

            all_data.append((scene_id, scene, scene_goal))
            #breakpoint()
            
        all_data = sorted(all_data, key=itemgetter(0))

        with open(filename_results, "w+") as f:
            f.write("scene_id\tsigma\tcol\tr\n")

        start = 0
        for i in tqdm(range(start, min(len(all_data),self.sample_size))):
            x = all_data[i]
            scene_id = x[0]
            scene = x[1]
            scene_goal = x[2]
            batch_split = torch.Tensor([0,scene.size(1)]).to(self.device).long()

            # print(scene)
            # print(i, scene_id)
            #visualize_scene(scene)
            #breakpoint()

            col, r = self.certify_scene(scene, scene_goal, batch_split)
            log(filename_results, scene_id, self.sigma, col, r)
            #breakpoint()


    def certify_scene(self, observed: torch.tensor, goals: torch.tensor, batch_split):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """

        #_, outputs = self.slstm(observed.clone(), goals.clone(), batch_split, n_predict=self.n_predict)
        #breakpoint()
        agents_count = len(observed[0])
        if agents_count <= 1: #solo agent
            return -2, -2

        #_sample_noise n0 -> pred
        # draw sample sof f(x+ epsilon)
        counts_selection = self.get_certify_counts(observed.clone(), goals.clone(), batch_split, self.n0)

        # use these samples to take a guess at the top class
        dominant_class = counts_selection.argmax().item()

        #_sample_noise n -> cert
        # draw more samples of f(x + epsilon)
        counts_estimation = self.get_certify_counts(observed.clone(), goals.clone(), batch_split, self.n)

        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[dominant_class].item()
        pABar = self._lower_confidence_bound(nA, self.n, self.alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            #breakpoint()
            return dominant_class, radius
        

    def get_certify_counts(self, observed: torch.tensor, goals: torch.tensor, batch_split, num):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """

        #use code of add_noise_on_perturbed

        #return [nocol, col]

        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            print_every = 100
            for n in range(num):
                outputs_perturbed, _ = self._sample_noise(observed, goals, batch_split)

                # Each Neighbors Distance to The Main Agent
                agents_count = len(observed[0]) #is first always interest ? YES all calulation are rel. to 1st
                distances = torch.sqrt(torch.sum((torch.square(outputs_perturbed[-self.pred_length:]
                                    - outputs_perturbed[-self.pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                                    self.pred_length, agents_count, 2))[:, 1:]), dim=2))

                # Score
                score = torch.min(distances).data
                
                #check if collision
                is_col = (score < self.collision_treshold)
                if is_col:
                    counts[1] += 1
                else : #no colision
                    counts[0] += 1

            return counts

    def _sample_noise(self, observed: torch.tensor, goals: torch.tensor, batch_split):
        """
        produce a output from a noisy version on input
        """
        with torch.no_grad():
            noise = observed.detach().clone().normal_(mean = 0, std = self.sigma)
            noise[:,1:,:] = 0 #modify only agent 0
            noise[:-self.time_noise_from_end,:,:] = 0 #add noise only on last timestep
            # if no_noise_on_last:
            #     noise[-1,:,:] = 0 
            noisy_observation = observed + noise

            #run the model
            _, outputs_perturbed = self.slstm(
                noisy_observation, goals, batch_split, n_predict=self.n_predict
            )
            return outputs_perturbed, noise    
    
    
    def predict_all(self, scenes: list, goals:list, filename_results:str, 
                    sigma:int, n0: int, 
                    alpha: float, n_predict:int =12):
        self.n0 = n0
        self.alpha = alpha
        self.n_predict = n_predict
        self.sigma = sigma 

        #random.shuffle(scenes)
        check_point_size = 50
        all_data = []


        #first preprocess the scenes
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

            all_data.append((scene_id, scene, scene_goal))
            #breakpoint()

        all_data = sorted(all_data, key=itemgetter(0))

        with open(filename_results, "w+") as f:
            f.write("scene_id\tsigma\tcol\tnoise_norm\n")

        start = 0
        all_pred = []
        for i in tqdm(range(start, min(len(all_data),self.sample_size))):
            x = all_data[i]
            scene_id = x[0]
            scene = x[1]
            scene_goal = x[2]
            batch_split = torch.Tensor([0,scene.size(1)]).to(self.device).long()

            # print(scene)
            # print(i, scene_id)
            #visualize_scene(scene)
            #breakpoint()

            col, pred, noise_norm  = self.predict_scene(scene, scene_goal, batch_split)
            log_pred(filename_results, scene_id, self.sigma, col, noise_norm)

            #do smth with the prediction
            if pred == None:
                all_pred.append(None)
                continue
                #classifier did astrain
            else :
                all_pred.append(pred)
            
            #breakpoint()
        return all_pred
        
    def predict_scene(self, observed: torch.tensor, goals: torch.tensor, batch_split):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        
    
        #_sample_noise n0 -> pred
        # draw sample sof f(x+ epsilon)
        self.slstm.eval()
        counts, prediction, noise_norm = self.get_predicted_counts(
            observed.clone(), goals.clone(), batch_split, self.n0
        )
        self.slstm.train()

        top2 = counts.argsort()[::-1] #return index of dominant class, then other, in this order
        dominant_class = counts[top2[0]]
        second = counts[top2[1]]
        if binom_test(dominant_class, self.n0, p=0.5) > self.alpha:
            return Smooth.ABSTAIN, None
        else:
            return top2[0], prediction[top2[0]], noise_norm[top2[0]]
        



    def get_predicted_counts(self, observed: torch.tensor, goals: torch.tensor, batch_split, num):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """

        #use code of add_noise_on_perturbed

        #return [nocol, col]

        smallest_pred = [None, None]
        smallest_l2_norm = np.ones(self.num_classes) * np.inf

        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            print_every = 100
            for n in range(num):
                outputs_perturbed, noise = self._sample_noise(observed, goals, batch_split)

                # Each Neighbors Distance to The Main Agent
                agents_count = len(observed[0]) #is first always interest ? YES all calulation are rel. to 1st
                distances = torch.sqrt(torch.sum((torch.square(outputs_perturbed[-self.pred_length:]
                                    - outputs_perturbed[-self.pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                                    self.pred_length, agents_count, 2))[:, 1:]), dim=2))

                # Score
                score = torch.min(distances).data
                
                #check if collision
                is_col = (score < self.collision_treshold)
                if is_col:
                    counts[1] += 1
                    if torch.norm(noise) < smallest_l2_norm[1]:
                        smallest_l2_norm[1] = torch.norm(noise)
                        smallest_pred[1] = outputs_perturbed
                else : #no colision
                    counts[0] += 1
                    if torch.norm(noise) <smallest_l2_norm[0]:
                        smallest_l2_norm[0] = torch.norm(noise)
                        smallest_pred[0] = outputs_perturbed

                # if (n%print_every) == 0:
                #     print("step : ", n)

            return counts, smallest_pred, smallest_l2_norm


    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

## UTILS##

def log(filename, scene_id, sigma, col, r):
    with open(filename,"a+") as f:
        f.write(str(scene_id) + "\t" + str(sigma) + "\t" + str(col) + "\t" + str(r) + "\n")

def log_pred(filename, scene_id, sigma, col, noise_norm):
    with open(filename,"a+") as f: 
        f.write(str(scene_id) + "\t" + str(sigma) + "\t" + str(col) + "\t" + str(noise_norm) + "\n")

def visualize_scene(scene, goal=None):
    for t in range(scene.shape[1]):
        path = scene[:, t]
        plt.plot(path[:, 0], path[:, 1])
    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])


    plt.show()
    plt.close()
