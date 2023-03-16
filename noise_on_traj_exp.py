import torch
import torch.nn as nn
import time
import os
import argparse
import random
import numpy as np



from trajnetbaselines.lstm.run import Trainer, draw_one_tensor, draw_two_tensor, prepare_data
from trajnetbaselines.lstm.utils import center_scene, random_rotation, save_log, calc_fde_ade
from trajnetbaselines.lstm.non_gridbased_pooling import HiddenStateMLPPooling, NN_LSTM, SAttention
from trajnetbaselines.lstm.lstm import LSTM



class AttackExperiment(Trainer):
    def __init__(self, model=None, criterion='L2', lr=None, barrier=1, show_limit=30,
                 device=None, batch_size=32, obs_length=9, pred_length=12, augment=False,
                 normalize_scene=False, save_every=1, start_length=0, obs_dropout=False,
                 sample_size = 70, reg_noise=0.5, reg_w=1, attacker_model_name='attacker.state',
                 discriminator_model_name='discriminator.state', collision_type = 'hard',
                 perturb_all = 'true', threads_limit = 4, speed_up='false', saving_name="", enable_thread='true',
                 output_dir='./out/',
                 sigmas = np.array([0.05])):
        super().__init__(model=model, criterion=criterion, lr=lr, barrier=barrier, show_limit=show_limit,
                 device=device, batch_size=batch_size, obs_length=obs_length, pred_length=pred_length, augment=augment,
                 normalize_scene=normalize_scene, save_every=save_every, start_length=start_length, obs_dropout=obs_dropout,
                 sample_size = sample_size, reg_noise=reg_noise, reg_w=reg_w, attacker_model_name=attacker_model_name,
                 discriminator_model_name=discriminator_model_name, collision_type = collision_type,
                 perturb_all = perturb_all, threads_limit = threads_limit, speed_up=speed_up, saving_name=saving_name, enable_thread=enable_thread,
                 output_dir=output_dir)
        self.sigmas = sigmas
        

    #overwritting existing method
    def attack_batch(self, xy, goals, batch_split, thread_index=None, scene_id = None):

        #breakpoint()
        local_model = self.models[thread_index]

        observed = xy[self.start_length:self.obs_length].clone()

        # prediction_truth: object = xy[self.obs_length:self.seq_length - 1].clone()
        # prediction_truth_for_numeric_stat: object = xy[self.obs_length:self.seq_length].clone()  ## CLONE
        first_observed = xy[self.start_length:self.obs_length].clone()
        # first_prediction_truth: object = xy[self.obs_length:self.seq_length].clone()

        temp_barrier = self.barrier
        # Setting the collision barrier on 20cm
        collision_done_barrier = 0.2 ####

        best_score_by_now = 10000
        best_loss_by_now = 15000
        best_observation_by_now = observed.clone()

        sf = nn.Softmax(dim=0)
        #sf = nn.functional.gumbel_softmax(dim=0, hard=True)

        agents_count = len(observed[0])
        w = torch.ones(self.pred_length, agents_count-1) / self.pred_length / (agents_count-1)
        w.requires_grad = True

        w_agent = torch.ones(agents_count-1) / (agents_count-1)
        w_agent.requires_grad = True

        w_frame = torch.ones(self.pred_length) / self.pred_length
        w_frame.requires_grad = True


        noise = torch.zeros(2 * self.obs_length, requires_grad=True)

        noise.data = self.clamp(noise.data, temp_barrier)


        optimizer = None
        if self.collision_type == 'soft':
            optimizer = torch.optim.Adam([noise] + [w], lr=self.lr)
        elif self.collision_type == 'hard':
            optimizer = torch.optim.Adam([noise], lr=self.lr)

        if agents_count <= 1:
            return -1, -1, -1

        target_agent_observed_path = observed[:, 0].reshape(-1)

        rel_outputs, outputs = local_model(observed.clone(), goals.clone(), batch_split, n_predict=12)
        num_agents = len(outputs.tolist()[0])

        best_outputs_perturbed = outputs.clone()

        counter_lim = 70
        counter = counter_lim
        while (counter != 0):

            # Adding Noise To Observation
            perturbed_path = target_agent_observed_path.clone().data
            if self.perturb_all == 'true':
                perturbed_path += noise
            else:
                perturbed_path[2:-2] += noise

            perturbed_observation = observed.clone()
            perturbed_observation[:self.obs_length, 0] = perturbed_path.view(-1, 2)

            temp_perturbed_observation = perturbed_observation.clone()

            # Models Output For Noisy Sample
            rel_outputs_perturbed, outputs_perturbed = local_model(perturbed_observation, goals.clone(), batch_split, n_predict=12)

            # Clear NaNs
            copy_outputs_perturbed = outputs_perturbed.clone()
            outputs_perturbed_nan = torch.isnan(outputs_perturbed)
            for i in range(len(outputs_perturbed)):
                for j in range(len(outputs_perturbed[0])):
                    if any(outputs_perturbed_nan[i, j].tolist()):
                        outputs_perturbed.data[i, j] = 10000
                        outputs_perturbed[i, j].detach()

            # Each Neighbors Distance to The Main Agent
            distances = torch.sqrt(torch.sum((torch.square(outputs_perturbed[-self.pred_length:]
                                  - outputs_perturbed[-self.pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                                  self.pred_length, agents_count, 2))[:, 1:]), dim=2))

            # Score
            score = torch.min(distances).data

            # Best Frame and Neighbor
            frame_index = (torch.argmin(distances) // (agents_count - 1)).data - self.pred_length
            neighbor_index =  (torch.argmin(distances) % (agents_count - 1) + 1).data

            # Collision Loss
            loss = None
            if self.collision_type == 'soft':
                w_sf = sf(w.view(-1)).view(self.pred_length, agents_count-1)
                A = (w_sf * torch.tanh(distances)).view(-1)
                loss = torch.sum(A[~torch.isnan(A)]) +  self.reg_noise * torch.norm(noise, 2, dim=0) - self.reg_w * torch.norm(w_sf.view(-1), 2, dim=0)

            elif self.collision_type == 'hard':
                first_agent_collision_point = outputs_perturbed[frame_index, 0]
                second_agent_collision_point = outputs_perturbed[frame_index, neighbor_index]
                loss = torch.norm(first_agent_collision_point - second_agent_collision_point, 2, dim=0)\
                    + self.reg_noise * torch.norm(noise, 2, dim=0)


            if loss < best_loss_by_now:
                #optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']  * (counter/70)
                # LR Decay and Counter Reset
                if loss < torch.tanh(torch.Tensor([3])) and self.collision_type == 'soft':
                    optimizer.param_groups[0]['lr'] = (1 - torch.max(sf(w))) / 10
                    if optimizer.param_groups[0]['lr'] < 0.001:
                        break
                # Saving The Best Records
                best_score_by_now = score.item()
                best_loss_by_now = loss.item()
                best_outputs_perturbed = copy_outputs_perturbed.clone()
                best_observation_by_now = temp_perturbed_observation.clone()
                best_frame = frame_index
                best_neighbor = neighbor_index
                if self.speed_up == 'true':
                    # check if collision has occured and perturbation is small enough
                    if (best_score_by_now < collision_done_barrier) and (torch.norm(noise, 2, dim=0)<0.05):
                        break
                else:
                    counter = counter_lim
            else:
                counter -= 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noise.data = self.clamp(noise.data, temp_barrier)



        while_end = time.time()
        all_samples = self.collision_counter + self.fail_counter
        perturbed_observation = best_observation_by_now.clone()

        #breakpoint()
        # for delta of outputs and perturb
        fde, ade = calc_fde_ade(output=outputs_perturbed[-self.pred_length:], ground_truth=outputs[-self.pred_length:])
        self.all_ade['delta'].append(ade)
        self.all_fde['delta'].append(fde)

        # Counting For Collision Ratio
        did_collide = False
        if best_score_by_now < collision_done_barrier:
            self.collision_counter += 1
            did_collide = True
            #only consider perturbation size when a collsion occured
            fde, ade = calc_fde_ade(output=perturbed_observation, ground_truth=observed)
            self.all_ade['observed'].append(ade)
            self.all_fde['observed'].append(fde)

            #NEW HERE
            self.add_noise_on_perturbed(
                perturbed_observation, goals, batch_split, 
                local_model, scene_id, collision_done_barrier = collision_done_barrier,
            )
        else:
            self.fail_counter += 1

        # Draw Images & Save Logs
        if self.count_draw < self.show_limit and num_agents >= 2:
            self.count_draw += 1

            if best_score_by_now < collision_done_barrier:
                save_log("Sample " + str(self.count_draw) + " Collided.", self.sample_status_address)
            else:
                save_log("Sample " + str(self.count_draw) + " Failed.", self.sample_status_address)

            real = torch.cat((first_observed[: self.obs_length], outputs[-self.pred_length:]))
            perturb = torch.cat((perturbed_observation[: self.obs_length], best_outputs_perturbed[-self.pred_length:]))

            if did_collide:
                filename = self.output_dir + str(scene_id) + '_altered_scene_ade: ' + str(
                    round(self.all_ade['observed'][-1], 3)) + '.png'
                filename_original = self.output_dir + str(scene_id) + '_original_scene.png'
            else:
                filename = self.output_dir + str(scene_id) + '_altered_scene.png'
                filename_original = self.output_dir + str(scene_id) + '_original_scene.png'

            draw_two_tensor(filename, real, perturb, best_outputs_perturbed[best_frame, 0].tolist()
                            , best_outputs_perturbed[best_frame, best_neighbor].tolist())
            draw_one_tensor(filename_original, real)


        if did_collide:
            # xy_per = torch.cat((perturbed_observation, prediction_truth_for_numeric_stat) )
            xy_per2 = torch.cat( (perturbed_observation, best_outputs_perturbed[-self.pred_length:] ) )
            self.save_real_data.append( (xy, goals) )

            # self.save_perturbed_data_groundtruth.append( (xy_per, goals) )
            self.save_perturbed_data_modelprediction.append( (xy_per2, goals) )

        return 0, 0, 0
    
    def add_noise_on_perturbed(
            self, perturbed_observation, goals, batch_split, 
            local_model, scene_id, N_noisy = 100, time_mod_from_end = 3, collision_done_barrier=0.2,
        ):
        #print("here")
        with torch.no_grad():
            _, out_original = local_model(
                    perturbed_observation, goals.clone(), batch_split, n_predict=12
                )
            for sigma in self.sigmas:
                sigma = np.round(sigma, 2)
                nb_col = 0
                once = False #disable figure drawing
                for n in range(N_noisy):
                    noise = perturbed_observation.detach().clone().normal_(mean = 0, std = sigma)
                    noise[:,1:,:] = 0 #modify only agent 0
                    noise[:-time_mod_from_end,:,:] = 0 #add noise only on last timestep
                    noisy_observation = perturbed_observation + noise

                    #run the model
                    _, outputs_perturbed = local_model(
                        noisy_observation, goals.clone(), batch_split, n_predict=12
                    )

                    # Each Neighbors Distance to The Main Agent
                    agents_count = len(perturbed_observation[0])
                    distances = torch.sqrt(torch.sum((torch.square(outputs_perturbed[-self.pred_length:]
                                        - outputs_perturbed[-self.pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                                        self.pred_length, agents_count, 2))[:, 1:]), dim=2))

                    # Score
                    score = torch.min(distances).data
                    
                    #check if collision
                    if (score < collision_done_barrier):
                        nb_col += 1
                    else : #no colision
                        if once : 
                            filename = self.output_dir + str(scene_id) + '_with_noise_' + str(sigma) + '.png'
                            
                            perturb = torch.cat((perturbed_observation[: self.obs_length], out_original[-self.pred_length:]))
                            real = torch.cat((noisy_observation[: self.obs_length], outputs_perturbed[-self.pred_length:]))
                            frame_index = (torch.argmin(distances) // (agents_count - 1)).data - self.pred_length
                            neighbor_index =  (torch.argmin(distances) % (agents_count - 1) + 1).data
                            
                            draw_two_tensor(
                                filename, real, perturb, outputs_perturbed[frame_index, 0].tolist()
                                    , outputs_perturbed[frame_index, neighbor_index].tolist()
                            )
    
                            once = False
                print(f"\n Number of colision with noise (sigma = {sigma}) added {N_noisy}"  
                    f"times : {nb_col} -> no_col_accuracy { (1-nb_col/N_noisy)*100:.2f}%")
                
                #record results
                with open(self.output_dir + "00_results" + ".txt" ,"a+") as f:
                    f.write(str(scene_id) + "\t" + str(sigma) + "\t" + str(nb_col) + "\n")
            

            


            
        


def main(epochs=10):

    #<------------- S-ATTack arguments ----------------#
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--barrier', default=1, type=float,
                        help='barrier for noise')
    parser.add_argument('--show_limit', default=50, type=int,
                        help='number of shown samples')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='s_lstm',
                        choices=('s_lstm', 'd_pool', 's_att'),
                        help='type of LSTM to train')
    parser.add_argument('--collision_type', default='hard',
                        choices=('hard', 'soft'),
                        help='method used for attack')
    parser.add_argument('--data_part', default='test',
                        choices=('test', 'train', 'val'),
                        help='data part to perform attack on')
    parser.add_argument('--models_path', default='trajnetbaselines/lstm/Target-Model/d_pool.state',
                        help='the directory of the model')
    parser.add_argument('--threads_limit', default=1, type=int,
                        help='number of checked samples')
    parser.add_argument('--enable_thread', default='true',
                        help='enable or disable multi-thread processing ')
    # -------------- S-ATTack arguments --------------->#


    # Trajnet++ arguments
    parser.add_argument('--loss_type', default='L2',
                        choices=('L2', 'collision'),
                        help='type of LSTM to train')
    parser.add_argument('--norm_pool', action='store_true',
                        help='normalize_pool (along direction of movement)')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling (only consider pedestrian in front along direction of movement)')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='augment scenes')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goal_path', default=None,
                        help='glob expression for goal files')
    parser.add_argument('--loss', default='L2',
                        help='loss function')
    parser.add_argument('--goals', action='store_true',
                        help='to use goals')
    parser.add_argument('--reg_noise', default=0.5, type=float,
                        help='noise regulizer weigth')
    parser.add_argument('--reg_w', default=1, type=float,
                        help='w regulizer weigth')
    parser.add_argument('--sample_size', default=70, type=int,
                        help='number of checked samples')
    parser.add_argument('--perturb_all', default='true',
                        choices=('true', 'false'),
                        help='perturb all the nodes or only ones in the middle')
    parser.add_argument('--speed_up', default='false',
                        choices=('true', 'false'),
                        help='speed up?')

    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ##Pretrain Pooling AE
    pretrain.add_argument('--load_pretrained_pool_path', default=None,
                          help='load a pickled model state dictionary of pool AE before training')
    pretrain.add_argument('--pretrained_pool_arch', default='onelayer',
                          help='architecture of pool representation')
    pretrain.add_argument('--downscale', type=int, default=4,
                          help='downscale factor of pooling grid')
    pretrain.add_argument('--finetune', type=int, default=0,
                          help='finetune factor of pretrained model')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='RNN hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=16,
                                 help='number of cells per side')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*',
                                 help='interaction module layer dims for gridbased pooling')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='pooling dimension')
    hyperparameters.add_argument('--embedding_arch', default='two_layer',
                                 help='interaction arch')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal dimension')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='attention mlp spatial dimension')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='attention mlp vel dimension')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background of pooling grid')
    hyperparameters.add_argument('--sample', default=1.0, type=float,
                                 help='sample ratio of train/val scenes')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for grid-based')
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='dont consider velocity in nn')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='neighbours to consider in DirectConcat')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iters in NMMP')
    hyperparameters.add_argument('--start_length', default=0, type=int,
                                 help='prediction length')
    hyperparameters.add_argument('--obs_dropout', action='store_true',
                                 help='obs length dropout')
    args = parser.parse_args()


    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    test_path = 'DATA_BLOCK/trajdata'
    args.path = 'DATA_BLOCK/' + args.path
    if args.data_part == 'test':
      test_scenes, test_goals = prepare_data(test_path, subset='/test/', sample=args.sample, goals=args.goals)
    else:
      test_scenes, test_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)

    # create model (Various interaction/pooling modules)
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                     mlp_dim_vel=args.vel_dim)
    elif args.type == 'd_pool':
        pool = NN_LSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 's_att':
        pool = SAttention(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)

    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim,
                 goal_flag=args.goals,
                 goal_dim=args.goal_dim)

    # Load model
    load_address = args.models_path
    print("Loading Model Dict from ", load_address)

    with open(load_address, 'rb') as f:
        checkpoint = torch.load(f)
    pretrained_state_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_state_dict, strict=False)

    '''model = trajnetbaselines.lstm.LSTMPredictor.load(load_address).model'''
    # Freeze the model
    for p in model.parameters():
        p.requires_grad = False

    print("Successfully Loaded")

    # trainer
    saving_name = str(args.type) + "-" + str(args.collision_type) + "-noise-"  + str(args.reg_noise) + "-w-" + str(args.reg_w) + "-barrier-" + str(args.barrier)

    #here new model
    sigmas = np.linspace(0.01, 0.13,7)
    trainer = AttackExperiment(model, lr=args.lr, device=args.device, barrier=args.barrier, show_limit=args.show_limit,
                    criterion=args.loss, collision_type = args.collision_type,
                    obs_length=args.obs_length, reg_noise = args.reg_noise, reg_w = args.reg_w,
                    pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
                    start_length=args.start_length, obs_dropout=args.obs_dropout,
                    sample_size = args.sample_size, perturb_all = args.perturb_all, threads_limit=args.threads_limit,
                    speed_up=args.speed_up, saving_name=saving_name, enable_thread=args.enable_thread,
                    output_dir=args.output,
                    sigmas=sigmas)
    trainer.attack(test_scenes, test_goals)
    trainer.numerical_stats()
    
if __name__ == "__main__":
    main()
    #create model

    #create AttackExperiment

    #modify the function to add noise
    pass
