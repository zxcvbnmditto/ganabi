import os
import numpy as np
import pickle


def resolve_pkl_file_name(
        game_type='Hanabi-Full',
        num_players=2,
        num_unique_agents=6,
        num_games=50):
    
    data_filename = game_type + "_" + str(num_players) + "_" \
            + str(num_unique_agents) + "_" + str(num_games) + ".pkl"
    return data_filename


def make_npy_file_name(game_num, game_steps):
    file_name = str(game_num) + "_" + str(game_steps) + ".npy"
    
    return file_name


# ./data/train/agent_1/obs
def make_npy_dir_name(default_path, agent_name, split_type, data_type):
    assert(data_type == 'obs' or data_type == 'act')
    assert(split_type == 'train' or split_type == 'validation' or split_type == 'test')

    path_name = os.path.join(default_path, split_type)
    path_name = os.path.join(path_name, agent_name)
    path_name = os.path.join(path_name, data_type)

    return path_name


def save_npy(data, path_name, file_name):
    file_name = os.path.join(path_name, file_name)
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name)) 

    np.save(file_name, data)


def read_pkl_data(file_name):
    try:
        raw_data = pickle.load(open(file_name, "rb"), encoding='latin1')

    except IOError:
        call("python create_data.py --datapath " + file_name, shell=True)
        raw_data = pickle.load(open(file_name, "rb"), encoding='latin1')
    
    return raw_data

def split_data(raw_data, split_ratio):
    train_data = {}
    validation_data = {}
    test_data = {}

    for agent in raw_data:
        train_idx = int(split_ratio[0] * len(raw_data[agent]))
        val_idx = int((split_ratio[0] + split_ratio[1]) * len(raw_data[agent]))
                    
        train_data[agent] = raw_data[agent][:train_idx]
        validation_data[agent] = raw_data[agent][train_idx:val_idx]   
        test_data[agent] = raw_data[agent][val_idx:]

    return train_data, validation_data, test_data