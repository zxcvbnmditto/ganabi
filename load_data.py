# Copy over code from earlier version.
# Load the data if it already exists, otherwise create it

from utils import parse_args
from utils import dir_utils
import gin
from subprocess import call
import pickle
import random
import numpy as np

@gin.configurable
class Dataset(object):
    @gin.configurable
    def __init__(self, 
            game_type='Hanabi-Full',
            num_players=2,
            num_unique_agents=6,
            num_games=150):

        self.game_type = game_type
        self.num_players = num_players
        self.num_unique_agents = num_unique_agents
        self.num_games = num_games

        self.train_data = {} # gameplay data given to model
        self.validation_data = {} # data not given to model, from same agents as train
        self.test_data = {} # data from agents totally unseen to model
        
    def read(self, raw_data):
        # split up raw_data into train, validation, and test
        test_agent = random.choice(list(raw_data.keys()))
    
        for agent in raw_data:
            if agent == test_agent:
                continue
            split_idx = int(0.9 * len(raw_data[agent]))
            self.train_data[agent] = raw_data[agent][:split_idx]
            self.validation_data[agent] = raw_data[agent][split_idx:]
            
        self.test_data[test_agent] = raw_data[test_agent]

    def get_size(self, batch_type='train'):
        if batch_type == 'train':
            data_bank = self.train_data
        elif batch_type == 'validation':
            data_bank = self.validation_data
        elif batch_type == 'test':
            data_bank = self.test_data
        
        size = 0
        for agent in data_bank:
            size += len(data_bank[agent])

        return size


    def generator(self, batch_size, batch_type='train'):
        if batch_type == 'train':
            data_bank = self.train_data
        elif batch_type == 'validation':
            data_bank = self.validation_data
        elif batch_type == 'test':
            data_bank = self.test_data
        
        # data_bank: [AgentName][num_games][0 = 
        #         obs_vec, 1 = act_vec][game_step][index into vec]
        # [agent_num][game_num][0 or 1][game_step][678 obs space]
        rand_id = random.choice(range(len(data_bank.keys())))
        agent = list(data_bank.keys())[rand_id]
        adhoc_games = [random.choice(list(data_bank[agent])) 
                for _ in range(batch_size)]
        game_lengths = [len(game[0]) for game in adhoc_games]
        
        # adhoc_games: [-->[[obs_act_vec],[obs_act_vec],...]<--game1, 
        #               -->[[obs_act_vec],[obs_act_vec],...]<--game2...]
        adhoc_games = [[adhoc_games[i][0][l] + adhoc_games[i][1][l] 
                       for l in range(game_lengths[i])] 
                       for i in range(batch_size)]
        
        # assemble generated agent observations and target actions
        agent_obs, agent_act = [], []
        for i in range(batch_size):
            game = random.choice(list(data_bank[agent]))
            step_num = random.randint(0, len(game[0])-1)
            agent_obs.append(game[0][step_num])
            agent_act.append(game[1][step_num])
        
        agent_obs = np.array(agent_obs)
        agent_act = np.array(agent_act)
        game_lengths = np.array(game_lengths)
        OBS_ACT_VEC_LEN = np.shape(agent_obs)[1] + np.shape(agent_act)[1]
        MAX_GAME_LEN = np.max(game_lengths)

        # convert nested uneven list of adhoc games into padded numpy array
        np_adhoc_games = np.zeros(shape=(batch_size, MAX_GAME_LEN, OBS_ACT_VEC_LEN)) 
        for game_num, game_len in enumerate(game_lengths):
            np_adhoc_games[game_num, :game_lengths[game_num], :] = \
                    np.asarray(adhoc_games[game_num])
        
        #FIXME: needs to return same_act
        return ([agent_obs, agent_act], agent_act)

def main(args):
    data = Dataset()
    args = parse_args.resolve_datapath(args,
        data.game_type,
        data.num_players,
        data.num_unique_agents,
        data.num_games)

    try:
        raw_data = pickle.load(open(args.datapath, "rb"), encoding='latin1')

    except IOError:
        call("python create_data.py --datapath " + args.datapath, shell=True)
        raw_data = pickle.load(open(args.datapath, "rb"), encoding='latin1')
    
    data.read(raw_data)

    # [np_adhoc_games, game_lengths, agent_obs], agent_act = data.generator(64, 'train')
    # import pdb; pdb.set_trace()
    return data


if __name__ == "__main__":
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)
    main(args)
