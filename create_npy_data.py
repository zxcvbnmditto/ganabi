# Copy over code from earlier version.
# Load the data if it already exists, otherwise create it

from utils import parse_args
from utils import dir_utils
import gin
from subprocess import call
import pickle
import random
import numpy as np
import keras
import os


def make_file_name(agent, game_num, game_steps):
    file_name = agent + '_' + str(game_num) + "_" + str(game_steps) + ".npy"
    
    return file_name

def make_dir_name(default_path, split_type, data_type):
    assert(data_type == 'obs' or data_type == 'act')
    assert(split_type == 'train' or split_type == 'validation' or split_type == 'test')

    path_name = os.path.join(default_path, split_type)
    path_name = os.path.join(path_name, data_type)

    return path_name

def save_npy(data, path_name, file_name):
    file_name = os.path.join(path_name, file_name)
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name)) 

    np.save(file_name, data)

    
class Dataset(object):
    def __init__(self,
            default_path,
            game_type='Hanabi-Full',
            num_players=2,
            num_unique_agents=6,
            num_games=150):

        self.game_type = game_type
        self.num_players = num_players
        self.num_unique_agents = num_unique_agents
        self.num_games = num_games    
        
        # Self Defined
        self.save_npy = True
        self.train_val_ratio = 0.9
        self.test_agent = None
        self.default_path = default_path
    
    def read_data(self, datapath):
        try:
            raw_data = pickle.load(open(datapath, "rb"), encoding='latin1')

        except IOError:
            call("python create_data.py --datapath " + datapath, shell=True)
            raw_data = pickle.load(open(datapath, "rb"), encoding='latin1')
        
        return raw_data


    def split_data(self, raw_data):
        train_data = {}
        validation_data = {}
        test_data = {}

        self.test_agent = random.choice(list(raw_data.keys()))
        for agent in raw_data:
            if agent == self.test_agent:
                continue
            split_idx = int(self.train_val_ratio * len(raw_data[agent]))
            train_data[agent] = raw_data[agent][:split_idx]
            validation_data[agent] = raw_data[agent][split_idx:]    
        test_data[self.test_agent] = raw_data[self.test_agent]

        return train_data, validation_data, test_data


    def refine_data(self, data, split_type):
        agent_names = list(data.keys())
        num_agent = len(agent_names)
        num_games = len(data[agent_names[0]])

        obs = [] 
        act = []
        game_lengths = np.zeros(shape=(num_agent, num_games))
        max_game_step = 0
        for i, agent in enumerate(data.keys()):
            obs.append([])
            act.append([])
            for game_id in range(0, num_games):
                tmp_obs = data[agent][game_id][0] 
                tmp_act = data[agent][game_id][1]
                obs[i].append(tmp_obs)
                act[i].append(tmp_act)
                
                game_length = np.shape(tmp_obs)[0]
                game_lengths[i, game_id] = game_length
                # print(np.shape(tmp_obs), np.shape(tmp_act))
                if game_length > max_game_step:
                    max_game_step = game_length
                
            if i == 0:
                obs_dim = np.shape(data[agent][0][0])[1]
                act_dim = np.shape(data[agent][0][1])[1]

        # Pad obs & act
        np_obs = np.zeros(shape=(num_agent, num_games, max_game_step, obs_dim))
        np_act = np.zeros(shape=(num_agent, num_games, max_game_step, act_dim))
        for i in range(num_agent):
            for j in range(0, num_games):
                game_length = int(game_lengths[i, j])
                np_obs[i, j, :game_length, :] = np.array(obs[i][j])
                np_act[i, j, :game_length, :] = np.array(act[i][j])

                if self.save_npy:
                    file_name = make_file_name(agent_names[i], j, game_length)
                    obs_path_name = make_dir_name(self.default_path, split_type, 'obs')
                    act_path_name = make_dir_name(self.default_path, split_type, 'act')

                    save_npy(np_obs[i, j, :, :], obs_path_name, file_name)
                    save_npy(np_act[i, j, :, :], act_path_name, file_name)

        return np_obs, np_act


    def setup(self, args):
        args = parse_args.resolve_datapath(args)

        # Read Raw Data using pickle
        print("----------------Parse Raw Data----------------")
        raw_data = self.read_data(self.default_path + '/Hanabi-Full_2_6_150.pkl')

        # Split the data into train / val / test
        print("----------------Split Data----------------")
        train_data, validation_data, test_data = self.split_data(raw_data)
        # import pdb; pdb.set_trace()
        
        print("----------------Refine Data----------------")
        train_obs, train_act = self.refine_data(train_data, 'train')
        validation_obs, validation_act = self.refine_data(validation_data, 'validation')
        test_obs, test_act = self.refine_data(test_data, 'test')


if __name__ == "__main__":
    # Hyper param
    default_path = '/home/james/Coding/ganabi/data'

    dataset = Dataset(default_path)
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)

    dataset.setup(args)