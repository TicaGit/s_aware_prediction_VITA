import numpy as np
import os
import argparse
import random
import torch
from operator import itemgetter
import matplotlib.pyplot as plt

from trajnetbaselines.lstm.lstm import LSTM, drop_distant
from trajnetbaselines.lstm.run import Trainer, draw_one_tensor, draw_two_tensor, prepare_data
from trajnetbaselines.lstm.utils import center_scene, random_rotation, save_log, calc_fde_ade, seperate_xy, is_stationary
from trajnetbaselines.lstm.non_gridbased_pooling import HiddenStateMLPPooling, NN_LSTM, SAttention

import trajnetplusplustools


def parse_args():
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
                        choices=('test', 'train', 'val', 'secret', 'secret_v2', 'synth'),
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
    return args

def main(epochs=10):

    args = parse_args()

    if args.sample < 1.0 or True: #FORCE RANDOM HERE
        torch.manual_seed("080819")
        random.seed(1)
        np.random.seed(1)

    #args.sample = None #TEST THAT TO DEACTIVATE RANDOM

    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    device = args.device
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    test_path = 'DATA_BLOCK/trajdata'
    args.path = 'DATA_BLOCK/' + args.path
    if args.data_part == 'test':
        test_scenes, test_goals = prepare_data(test_path, subset='/test/', sample=args.sample, goals=args.goals)
    elif args.data_part == 'train':
        test_scenes, test_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals)
    elif args.data_part == 'secret': #NEW - UNTRACKED - ONLY HERE LOCALLY
        test_scenes, test_goals = prepare_data(args.path, subset='/test_private/', sample=args.sample, goals=args.goals)
    elif args.data_part == 'secret_v2': #NEW - UNTRACKED - ONLY HERE LOCALLY
        test_scenes, test_goals = prepare_data(args.path, subset='/test_private_v2/', sample=args.sample, goals=args.goals)
    elif args.data_part == 'synth': #NEW
        test_scenes, test_goals = prepare_data(args.path, subset='/synth_data/', sample=args.sample, goals=args.goals)
        

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

    #torch.save(model.state_dict(), "evaluator/copy_of_model/lstm_d_pool.pkl")

    '''model = trajnetbaselines.lstm.LSTMPredictor.load(load_address).model'''
    # Freeze the model
    for p in model.parameters():
        p.requires_grad = False

    print("Successfully Loaded")


    ### NEW ###
    obs_length = args.obs_length    #9
    pred_length = args.pred_length  #12
    collision_treshold = 0.2 #same as in run.py #orignal saeed : 0.2

    ###############################
    ### load + preproc all data ###
    ###############################
    all_data = []
    modif = 0
    for i, (filename, scene_id, paths) in enumerate(test_scenes):
        
        paths_before = paths
        if args.data_part == 'secret':
            #drop agent > obs lenght
            paths = drop_post_obs(paths, obs_length)
        

        if paths != paths_before:
            modif += 1
            #breakpoint()

        scene = trajnetplusplustools.Reader.paths_to_xy(paths) # Now T_obs x N_agent x 2

        #filtered_tensor = scene[~torch.any(scene[:obs_length].isnan(),dim=0)]
        #scene = scene[:obs_length] #cut at obs lenght

        
        ## get goals
        if test_goals is not None:
            scene_goal = np.array(test_goals[filename][scene_id])
        else:
            scene_goal = np.array([[0, 0] for path in paths])

        ## DO Drop Distant : handle solo agent with col = "None"
        drop_dist = True
        if drop_dist:
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

        scene = torch.Tensor(scene).to(device)
        scene_goal = torch.Tensor(scene_goal).to(device)

        ## DONT remove stationnary
        remove_static = False
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
    #3146 tot lenght of private secret test set

    #13718 training data

    print(f"scene with post obs remooved : {modif}")


    #all_data = all_data[0:2]

    ##########
    ## iter ##
    ##########
    filename = "out/no_noise/dummy.txt"
    with open(filename,"w+") as f:
        f.write("scene_id"+ "\t" + "col" + "\t" +"col_trajnetpp" + "\t" + "ade" + "\t" + "fde" + "\n")
    tot = 0
    diff_col = 0
    for iter ,data in enumerate(all_data):
        tot += 1

        print(iter)#, end="\r")
        
        scene_id = data[0]
        scene = data[1]
        #breakpoint()
        scene_goal = data[2]

        batch_split = torch.Tensor([0,scene.size(1)]).to(device).long()

        _, model_pred = model(scene[:obs_length].clone(),  
                                scene_goal, 
                                batch_split, 
                                n_predict=pred_length)
        breakpoint()

        # model_pred_orig = model_pred.clone().detach()

        # if torch.isnan(model_pred[:,0,:]).any():
        #     breakpoint() #never :)
        
        #necessary : if any(distances) is nan, then torch.min = nan -> never a colision
        pred_nan = torch.isnan(model_pred)
        for j in range(len(model_pred)):
            for k in range(len(model_pred[0])):
                if any(pred_nan[j, k].tolist()):
                    model_pred.data[j, k] = 10000
                    model_pred[j, k].detach()

        #model_pred = torch.cat((scene[:obs_length], model_pred[-pred_length:]))

        #model_pred = torch.cat((model_pred, torch.zeros(2,model_pred.shape[1],2))) #to draw

        # Each Neighbors Distance to The Main Agent
        agents_count = len(model_pred[0]) #all calulation are rel. to 1st
        if agents_count <= 1: #solo agents
            with open(filename,"a") as f:
                f.write(str(scene_id) + "\t" + str(-1) + "\t" + str(-1) + "\t" + str(-1) + "\t" + str(-1) + "\n")
            continue

        distances = torch.sqrt(torch.sum((torch.square(model_pred[-pred_length:]
                            - model_pred[-pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                            pred_length, agents_count, 2))[:, 1:]), dim=2))
        
        # Score
        score = torch.min(distances).data

        col = int(score < collision_treshold)

        #like in colision in trajnet_eval
        ## Collision in Predictions
        # [Col-I] only if neighs in gt = neighs in prediction
        num_gt_neigh = scene.shape[1] - 1
        num_predicted_neigh = model_pred.shape[1] - 1 
        if num_gt_neigh != num_predicted_neigh:
            breakpoint() #nerver !



        #like in evaluator 
        # primary_tracks_all = [t for t in self.scenes_sub[i][0] if t.scene_id == scene]
        # neighbours_tracks_all = [[t for t in self.scenes_sub[i][j] if t.scene_id == self.scenes_id_gt[i]] for j in range(1, len(self.scenes_sub[i]))]


        col_trajnetpp = 0
        inter_parts = 2 #num interpolation points
        only_pred = model_pred[-obs_length:]
        for j in range(1, only_pred.shape[1]):# all neighbours (0 is main)

            for i in range(only_pred.shape[0] - 1): #num timesteps (11-1)
                p1 = [only_pred[i,0,0], only_pred[i,0,1]] #main agent, time i
                p2 = [only_pred[i+1,0,0], only_pred[i+1,0,1]] #main agent, time i + 1

                p3 = [only_pred[i,j,0], only_pred[i,j,1]] #neighbours j, time i
                p4 = [only_pred[i+1,j,0], only_pred[i+1,j,1]] #neighbours j, time i +1
                if np.min(np.linalg.norm(getinsidepoints(p1, p2, inter_parts) - getinsidepoints(p3, p4, inter_parts), axis=0)) \
                <= collision_treshold:
                    col_trajnetpp = 1
        
        if col != col_trajnetpp:
            #breakpoint() #some diffenrece !!
            diff_col += 1



        #orig scene treat without nan replacement
        # distances_orig = torch.sqrt(torch.sum((torch.square(model_pred_orig[-pred_length:]
        #                     - model_pred_orig[-pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
        #                     pred_length, agents_count, 2))[:, 1:]), dim=2))
        # score_orig = torch.min(distances_orig).data

        # col_orig = int(score_orig < collision_treshold)

        # if col != col_orig:
        #     breakpoint()

        #metric
        fde, ade = calc_fde_ade(model_pred[-pred_length:], scene[-pred_length:])

        with open(filename,"a") as f:
            f.write(str(scene_id) + "\t" + str(col) + "\t" + str(col_trajnetpp) + "\t" + str(ade) + "\t" + str(fde) + "\n")

        #breakpoint()

        # if i == 119:
        #     visualize_scene(model_pred_orig)
        #     visualize_scene(model_pred)
        #     breakpoint()
        #     draw_two_tensor("out/no_noise/first_display", model_pred, scene)
        
    print(tot)
    print(f"Difference in colision between methods : {diff_col}")

def visualize_scene(scene, goal=None):
    for t in range(scene.shape[1]):
        path = scene[:, t]
        plt.plot(path[:, 0], path[:, 1], label = t)
    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])
    plt.legend()
    plt.show()

## from trajnet_evaluator l.280
## drop pedestrians that appear post observation
# expect other scene representation
def drop_post_obs(ground_truth, obs_length):
    obs_end_frame = ground_truth[0][obs_length].frame
    ground_truth = [track for track in ground_truth if track[0].frame < obs_end_frame]
    return ground_truth

## from trajnet++ mertrics (site-package)
# expect other scene representation
def getinsidepoints(p1, p2, parts=2):
    """return: equally distanced points between starting and ending "control" points"""

    return np.array((np.linspace(p1[0], p2[0], parts + 1),
                        np.linspace(p1[1], p2[1], parts + 1)))



if __name__ == '__main__':
    main()