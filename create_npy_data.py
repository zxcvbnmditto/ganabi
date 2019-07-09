# Please Ignore this file for now. 
# The functionality has been integrated into create_data.py

from utils import parse_args
from utils import dir_utils
from utils import npy_utils
from collections import defaultdict
import gin
from subprocess import call
import pickle
import random
import numpy as np
import keras
import os


@gin.configurable
class Dataset(object):
    @gin.configurable
    def __init__(self,
                 args,
                 game_type='Hanabi-Full',
                 num_players=2,
                 num_unique_agents=6,
                 num_games=150,
                 split_ratio= [0.1, 0.8, 0.1]):

        self.default_path = args.datadir
        self.split_ratio = split_ratio
        self.pkl_filename = npy_utils.resolve_pkl_file_name(game_type, num_players,num_unique_agents, num_games)


    def split_data(self, raw_data):
        train_data = {}
        validation_data = {}
        test_data = {}

        for agent in raw_data:
            train_idx = int(self.split_ratio[0] * len(raw_data[agent]))
            val_idx = int((self.split_ratio[0] + self.split_ratio[1]) * len(raw_data[agent]))
                        
            train_data[agent] = raw_data[agent][:train_idx]
            validation_data[agent] = raw_data[agent][train_idx:val_idx]   
            test_data[agent] = raw_data[agent][val_idx:]

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

                obs_path_name = npy_utils.make_npy_dir_name(self.default_path, agent_names[i], split_type, 'obs')
                act_path_name = npy_utils.make_npy_dir_name(self.default_path, agent_names[i], split_type, 'act')  
                file_name = npy_utils.make_npy_file_name(j, game_length)
                
                npy_utils.save_npy(np_obs[i, j, :, :], obs_path_name, file_name)
                npy_utils.save_npy(np_act[i, j, :, :], act_path_name, file_name)

        return np_obs, np_act


    def setup(self):
        # Read Raw Data using pickle
        print("----------------Parse Raw Data----------------")
        raw_data = npy_utils.read_pkl_data(os.path.join(self.default_path, self.pkl_filename))

        # Split the data into train / val / test
        print("----------------Split Data----------------")
        train_data, validation_data, test_data = self.split_data(raw_data)
        # import pdb; pdb.set_trace()
        
        print("----------------Refine Data----------------")
        train_obs, train_act = self.refine_data(train_data, 'train')
        validation_obs, validation_act = self.refine_data(validation_data, 'validation')
        test_obs, test_act = self.refine_data(test_data, 'test')


if __name__ == "__main__":
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)

    dataset = Dataset(args.datadir)
    dataset.setup()