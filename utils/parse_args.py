import os
import gin
import argparse
import random

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                        default='full_gan',
                        help='which part of ablation study or baseline to run')

    parser.add_argument('--modedir',
                        default='./modes')

    parser.add_argument('--configpath',
                        help='gin config file path')

    parser.add_argument('--datadir',
                        default='./data/')
    
    parser.add_argument('--datapath',
                        help='set automatically; specify only if data in unusual loc')

    parser.add_argument('--expertdir',
                        default='./experts')

    parser.add_argument('--ckptdir')

    parser.add_argument('--resultdir')

    parser.add_argument('--outdir',
                        default='./output/')

    parser.add_argument('--pklfile',
                        default='Hanabi-Full_2_6_150.pkl',
                        help="Set as the name of the raw pickle file. Used to "
                             "create numpy files.")
    
    parser.add_argument('--agentname',
                        default='rainbow_agent_1',
                        help="Set as the name of the agent to be train on")

    parser.add_argument('-newrun',
                        action='store_true',
                        help="If specified, creates a directory inside the output "
                             "directory (specified with --outdir), with a "       
                             "checkpoint and results directory inside it, plus a "
                             "copy of the gin config files. The run ID is the "   
                             "next available number.")

    parser.add_argument('-newnpy',
                        action='store_true',
                        help="If specified, creates three directories, train, "
                             "validation, and test, inside the ./data directory"
                             " (specified with --datadir). In each directory"
                             " create two directories obs and act. Numpy files "
                             "is stored under obs and act, where each file"
                             " contains the information of a full game of"
                             "either observation or action. Refer moew to the"
                             " create_npy_data.py script.")

    args = parser.parse_args()
    return args

def resolve_datapath(args,
        game_type='Hanabi-Full',
        num_players=2,
        num_unique_agents=6,
        num_games=50):
    
    if args.datapath == None:
        data_filename = game_type + "_" + str(num_players) + "_" \
                + str(num_unique_agents) + "_" + str(num_games) + ".pkl"
        args.datapath = os.path.join(args.datadir, data_filename)

    return args

def resolve_configpath(args):
    if args.configpath == None:
        config_filename = args.mode + ".config.gin"
        args.configpath = os.path.join(args.modedir, config_filename)
    
    return args


def resolve_agentname(args):
    if args.agentname == None:
        agents_dir = os.path.join(os.path.join(args.datadir, 'train'))
        agentname = [name for name in os.listdir(agents_dir) if os.path.isdir(name)]
        args.agentname = random.choice(agentname)
    
    return args