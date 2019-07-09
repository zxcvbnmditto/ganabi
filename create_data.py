# this script creates data using python 2 and rainbow agents

from utils import dir_utils
from utils import parse_args
from utils import npy_utils
from collections import defaultdict
import pickle
import sys
sys.path.insert(0, './hanabi-env') #FIXME
import rl_env
import gin
import os
import tensorflow as tf
import importlib
import numpy as np

def import_agents(expertdir, agent_config):
    available_agents = {}
    sys.path.insert(0, expertdir)

    for agent_filename in os.listdir(expertdir):
        if 'agent' not in agent_filename:
            continue 
        agent_name = os.path.splitext(agent_filename)[0]
        agent_module = importlib.import_module(agent_name)
        available_agents[agent_name] = agent_module.Agent(agent_config)

    return available_agents

def one_hot_vectorized_action(agent, num_moves, obs):
    action = agent.act(obs)
    act_vec_len = num_moves
    one_hot_vector = [0]*act_vec_len
    action_idx = obs['legal_moves_as_int'][obs['legal_moves'].index(action)]
    one_hot_vector[action_idx] = 1

    return one_hot_vector, action

# FIXME: config.gin is not working 
@gin.configurable
class Dataset(object):
    @gin.configurable
    def __init__(self, args,
                game_type='Hanabi-Full',
                num_players=2,
                num_unique_agents=1,
                num_games=100,
                split_ratio=[0.1, 0.8, 0.1],
                save_npy=True):

        # Old
        self.game_type = game_type
        self.num_players = num_players
        self.num_unique_agents = num_unique_agents
        self.num_games = num_games

        # FIXME: More class attribute refactoring
        # New
        self.split_ratio = split_ratio
        self.save_npy = save_npy

        self.default_path = args.datadir
        self.max_game_steps = 0
        self.obs_dim = -1
        self.act_dim = -1
        
        # Old
        self.environment = rl_env.make(game_type, num_players=self.num_players)        
        self.agent_config = {
                'players': self.num_players,
                'num_moves': self.environment.num_moves(),
                'observation_size': self.environment.vectorized_observation_shape()[0]}
        self.available_agents = import_agents(args.expertdir, self.agent_config)


    # Original Method: save data as pickle file
    def create_data(self):
        raw_data = defaultdict(list)

        for playing_agent in self.available_agents.keys():
            for game_num in range(self.num_games):
                raw_data[playing_agent].append([[],[]])
                observations = self.environment.reset()
                game_done = False

                while not game_done:
                    for agent_id in range(self.num_players):
                        observation = observations['player_observations'][agent_id]
                        action_vec, action = one_hot_vectorized_action(
                                self.available_agents[playing_agent],
                                self.environment.num_moves(),
                                observation)
                        raw_data[playing_agent][game_num][0].append(
                                observation['vectorized'])
                        raw_data[playing_agent][game_num][1].append(action_vec)

                        if observation['current_player'] == agent_id:
                            assert action is not None
                            current_player_action = action
                        else:
                            assert action is None

                        observations, _, game_done, _ = self.environment.step(
                                current_player_action)
                        if game_done:
                            break

        return raw_data

    # New Method: save data as numpy file
    def create_data_v2(self):
        obs = defaultdict(list)
        act = defaultdict(list)
        game_steps = defaultdict(list)

        for playing_agent in self.available_agents.keys():
            for game_num in range(self.num_games):
                obs[playing_agent].append([])
                act[playing_agent].append([])

                observations = self.environment.reset()
                game_done = False
                step = 0
                while not game_done:
                    for agent_id in range(self.num_players):
                        step += 1
                        observation = observations['player_observations'][agent_id]
                        action_vec, action = one_hot_vectorized_action(
                                self.available_agents[playing_agent],
                                self.environment.num_moves(),
                                observation)
                        
                        obs[playing_agent][game_num].append(
                                observation['vectorized'])
                        act[playing_agent][game_num].append(action_vec)
                        
                        if self.obs_dim == -1:
                            self.obs_dim = np.shape(observation['vectorized'])[0]
        
                        if self.act_dim == -1:
                            self.act_dim = np.shape(action_vec)[0]

                        if observation['current_player'] == agent_id:
                            assert action is not None
                            current_player_action = action
                        else:
                            assert action is None

                        observations, _, game_done, _ = self.environment.step(
                                current_player_action)
                        if game_done:
                            break
                
                if step > self.max_game_steps:
                    self.max_game_steps = step

                game_steps[playing_agent].append(step)
        
        return obs, act, game_steps


    def pad_then_save_data(self, obs, act, steps, split_type):
        agent_names = list(obs.keys())
        num_agent = len(agent_names)
        num_games = len(obs[agent_names[0]])
        
        # Pad obs & act
        np_obs = np.zeros(shape=(num_agent, num_games, self.max_game_steps, self.obs_dim))
        np_act = np.zeros(shape=(num_agent, num_games, self.max_game_steps, self.act_dim))
        for i in range(num_agent):
            agent_name = agent_names[i]
            for j in range(0, num_games):
                game_length = steps[agent_name][j]
                np_obs[i, j, :game_length, :] = np.array(obs[agent_name][j])
                np_act[i, j, :game_length, :] = np.array(act[agent_name][j])

                obs_path_name = npy_utils.make_npy_dir_name(self.default_path, agent_name, split_type, 'obs')
                act_path_name = npy_utils.make_npy_dir_name(self.default_path, agent_name, split_type, 'act')  
                file_name = npy_utils.make_npy_file_name(j, game_length)
                
                npy_utils.save_npy(np_obs[i, j, :, :], obs_path_name, file_name)
                npy_utils.save_npy(np_act[i, j, :, :], act_path_name, file_name)

        return np_obs, np_act


def main(args):
    data_creator = Dataset(args)

    # TODO: One method to create both pkl and npy data
    if not data_creator.save_npy: # Old: use pickle file
        raw_data = data_creator.create_data()
        pickle.dump(raw_data, open(args.datapath, "wb"))
    
    else: # New: use npy file
        obs, act, steps = data_creator.create_data_v2()

        # FIXME: split_ratio location
        train_obs, validation_obs, test_obs = npy_utils.split_data(obs, data_creator.split_ratio)
        train_act, validation_act, test_act = npy_utils.split_data(act, data_creator.split_ratio)
        train_steps, validation_steps, test_steps = npy_utils.split_data(steps, data_creator.split_ratio)
        
        data_creator.pad_then_save_data(train_obs, train_act, train_steps, 'train')
        data_creator.pad_then_save_data(validation_obs, validation_act, validation_steps, 'validation')
        data_creator.pad_then_save_data(test_obs, test_act, test_steps, 'test')


if __name__ == '__main__':
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)
    args = dir_utils.resolve_npy_directory(args)

    # FIXME: Config file not working
    # gin.parse_config_file('naive_mlp.config.gin')
    main(args)
    
