import numpy as np
import os
import argparse
import random
import torch
from operator import itemgetter

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
                        choices=('test', 'train', 'val', 'secret'),
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


    ### NEW ###
    obs_length = args.obs_length    #9
    pred_length = args.pred_length  #12
    collision_treshold = 0.2 #same as in run.py

    ###############################
    ### load + preproc all data ###
    ###############################
    all_data = []
    for i, (filename, scene_id, paths) in enumerate(test_scenes):

        scene = trajnetplusplustools.Reader.paths_to_xy(paths) # Now T_obs x N_agent x 2
        
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



    ##########
    ## iter ##
    ##########
    filename = "out/no_noise/no_noise_train.txt"
    with open(filename,"w+") as f:
        f.write("scene_id"+ "\t" + "col" + "\t" + "ade" + "\t" + "fde" + "\n")
    tot = 0
    for i,data in enumerate(all_data):

        print(i, end="\r")
        
        scene_id = data[0]
        scene = data[1]
        scene_goal = data[2]

        batch_split = torch.Tensor([0,scene.size(1)]).to(device).long()

        _, model_pred = model(scene[:obs_length],  
                                scene_goal, 
                                batch_split, 
                                n_predict=pred_length)
        breakpoint()
        model_pred = torch.cat((scene[:obs_length], model_pred[-pred_length:]))

        # Each Neighbors Distance to The Main Agent
        agents_count = len(model_pred[0]) #all calulation are rel. to 1st
        if agents_count <= 1: #solo agents
            with open(filename,"a") as f:
                f.write(str(scene_id) + "\t" + str(None) + "\t" + str(None) + "\t" + str(None) + "\n")
            continue

        distances = torch.sqrt(torch.sum((torch.square(model_pred[-pred_length:]
                            - model_pred[-pred_length:, 0].repeat_interleave(agents_count, 0).reshape(
                            pred_length, agents_count, 2))[:, 1:]), dim=2))
        
        # Score
        score = torch.min(distances).data

        is_col = (score < collision_treshold)
        col = 0
        if is_col:
            col = 1

        fde, ade = calc_fde_ade(model_pred[-pred_length:], scene[-pred_length:])

        with open(filename,"a") as f:
            f.write(str(scene_id) + "\t" + str(col) + "\t" + str(ade) + "\t" + str(fde) + "\n")
        
        tot += 1
    print(tot)






if __name__ == '__main__':
    main()