import torch
import numpy as np
import math
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import random
from operator import itemgetter


import trajnetplusplustools
from trajnetbaselines.lstm.lstm import drop_distant
from trajnetbaselines.lstm.run import draw_one_tensor



class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, 
                 slstm: torch.nn.Module, 
                 sigma: float = 0.01, 
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
        self.sigma = sigma
        self.device = device
        self.sample_size = sample_size
        self.time_noise_from_end = time_noise_from_end
        self.pred_length = pred_length
        self.collision_treshold = collision_treshold

        self.num_classes = 2 #binary, col or no_col

    def random_x(self, x):
      """
      sort of coding
      """
      y = 5
      mod = 1000000007
      for i in range(y):
        x = ((113 * x) + 81) % mod
      return x


    def certify_all(self, scenes: list, goals:list, filename_results:str, n0: int, n: int, alpha: float,
                    batch_size: int, n_predict:int =12):
        """
        cerfify
        """
        self.n0 = n0
        self.n = n
        self.alpha = alpha
        self.batch_size = batch_size #useless !?
        self.n_predict = n_predict

        random.shuffle(scenes)
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
            f.write("scene_id \t sigma \t col \t r \n")


        for i in tqdm(range(min(len(all_data),self.sample_size))):
            x = all_data[i]
            scene_id = x[0]
            scene = x[1]
            scene_goal = x[2]
            batch_split = torch.Tensor([0,scene.size(1)]).to(self.device).long()

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

        _, outputs = self.slstm(observed.clone(), goals.clone(), batch_split, n_predict=self.n_predict)
        #breakpoint()


        #_sample_noise n0 -> pred
        # draw sample sof f(x+ epsilon)
        counts_selection = self._sample_noise(observed.clone(), goals.clone(), batch_split, self.n0)

        # use these samples to take a guess at the top class
        dominant_class = counts_selection.argmax().item()

        #_sample_noise n -> cert
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(observed.clone(), goals.clone(), batch_split, self.n)

        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[dominant_class].item()
        pABar = self._lower_confidence_bound(nA, self.n, self.alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            #breakpoint()
            return dominant_class, radius
        
    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
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
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, observed: torch.tensor, goals: torch.tensor, batch_split, num) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """

        #use code of add_noise_on_perturbed

        #return [p(nocol), p(col)]

        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            print_every = 100
            for n in range(num):
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

                # Each Neighbors Distance to The Main Agent
                agents_count = len(observed[0]) #is first always interest ? 
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

                if (n%print_every) == 0:
                    print("step : ", n)

            return counts


    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
def log(filename, scene_id, sigma, col, r):
    with open(filename,"a+") as f:
        f.write(str(scene_id) + "\t" + str(sigma) + "\t" + str(col) + "\t" + str(r) + "\n")
