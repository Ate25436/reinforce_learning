import argparse
import random
from collections import Counter

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from custom_envs.envs.field import TCGEnv
from custom_envs.envs.tcg_env import TCGEnv_v2
from matplotlib import pyplot as plt
from scipy import stats as st
from stable_baselines3 import DQN
from tqdm import tqdm

from tools.stop_watch import stop_watch


def make_args():
    parser = argparse.ArgumentParser(description='DQN for TCG')
    parser.add_argument('--model_name', type=str, default='base_model', help='model name')
    parser.add_argument('--iter_num', type=int, default=10, help='number of iterations')
    parser.add_argument('--deck_name', type=str, default='', help='deck name')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'rule'],
                        help='mode to test the model, random or rule')
    return parser.parse_args()
class BattleAgents:

    def __init__(self, model_name):
        self.model_path = 'models/' + model_name

    def battle_and_write(self):
        env = TCGEnv()
        model = DQN.load(self.model_path)

        obs, _ = env.reset()
        env.render()
        done = False
        i = 0
        winner = ""
        with open('log.txt', 'w') as f:
            f.write('')
            while not done:
                agent = env.TurnPlayer
                switch_agent = 'agent_1' if agent == 'agent_0' else 'agent_0'
                save_env_before = env.t_save_env()
                action, _ = model.predict(obs[agent])
                action_dict = {agent: action, switch_agent: 0}
                obs, rewards, terminated, _, _ = env.step(action_dict)
                save_env_after = env.t_save_env()

                if save_env_before != save_env_after:
                    f.write('-' * 16 + str(action) + '-' * 16 + '\n')
                    f.write(str(rewards))
                    f.write(env.env_to_text() + '\n')
                    for key in save_env_before.keys():
                        if save_env_before[key] != save_env_after[key] and key == 'TurnPlayer':
                            f.write(key + '; ')
                        elif save_env_before[key] != save_env_after[key]:
                            change_agents = []
                            for agent in env.agents:
                                if save_env_before[key][agent] != save_env_after[key][agent]:
                                    change_agents.append(agent)
                            f.write(key + ': ' + ', '.join(change_agents)  + '; ')
                    f.write('\n\n')
                    i += 1
                done = terminated[agent]
                if done:
                    winner = "agent_0" if rewards["agent_0"] == 1.0 else "agent_1"
            print(i)
            return winner

    def battle(self):
        action_list = []
        env = TCGEnv()
        model = DQN.load(self.model_path)

        obs, _ = env.reset()
        env.render()
        done = False
        i = 0
        winner = ""
        while True:
            agent = env.TurnPlayer
            switch_agent = 'agent_1' if agent == 'agent_0' else 'agent_0'
            save_env_before = env.t_save_env()
            action, _ = model.predict(obs[agent])
            action_list.append(int(action))
            action_dict = {agent: action, switch_agent: 0}
            obs, rewards, terminated, _, _ = env.step(action_dict)
            save_env_after = env.t_save_env()
            if save_env_before != save_env_after:
                i += 1
            done = terminated[agent]
            if done:
                winner = "agent_0" if rewards["agent_0"] == 1.0 else "agent_1"
                break
        print(i)
        return winner, action_list

class BattleRandomAgent:

    def __init__(self, model_name='base_model', deck_name=''):
        self.model_name = model_name
        self.model_path = 'models/' + model_name
        self.deck_name = deck_name
        if deck_name != '':
            self.deck = pd.read_pickle('decks/' + deck_name + '.pkl')
        else:
            self.deck = []
        

    def battle_with_random(self):
        env = TCGEnv()
        model = DQN.load(self.model_path)
        random_agent = random.choice(['agent_0', 'agent_1'])
        model_agent = 'agent_1' if random_agent == 'agent_0' else 'agent_0'
        info = {'deck':{model_agent: self.deck}}
        obs, _ = env.reset(options=info)
        env.render()
        done = False
        i = 0
        winner = ""
        while True:
            agent = env.TurnPlayer
            switch_agent = 'agent_1' if agent == 'agent_0' else 'agent_0'
            if agent == random_agent:
                action = random.randrange(0, 40)
            else:
                action, _ = model.predict(obs[agent])
            action_dict = {agent: action, switch_agent: 0}
            obs, rewards, terminated, _, _ = env.step(action_dict)
            i += 1
            done = terminated[agent]
            if done:
                winner = "agent_0" if rewards["agent_0"] == 1.0 else("agent_1" if rewards["agent_1"] == 1.0 else "draw")
                break
        print(i)
        print('random_agent:', random_agent)
        print('winner:', winner)
        if winner == random_agent:
            winner = 'random'
        else:
            winner = 'model'
        return winner, i

    def calculate_win_rate_with_random(self, iter_num=100):
        win = 0
        turn_sum = 0
        for _ in range(iter_num):
            winner, i = self.battle_with_random()
            turn_sum += i
            if winner == 'model':
                win += 1
        return win / iter_num, turn_sum / iter_num

class BattleRuleAgent:

    def __init__(self, model_name='base_model', deck_name=''):
        self.model_name = model_name
        self.model_path = 'models/' + model_name
        self.deck_name = deck_name
        if deck_name != '':
            self.deck = pd.read_pickle('decks/' + deck_name + '.pkl')
        else:
            self.deck = []
        

    def model_vs_rule(self):
        env = TCGEnv_v2()
        model = DQN.load(self.model_path)
        if self.deck != []:
            obs, _ = env.reset(deck=self.deck)
        else:
            obs, _ = env.reset()
        env.render()
        terminated = False
        i = 0
        winner = ""
        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, _, _ = env.step(action)
            if terminated:
                if reward == 1.0:
                    winner = 'model'
                elif reward == -1.0:
                    winner = 'rule'
                else:
                    winner = 'draw'
                break
            i += 1
        return winner, i
        
    def calculate_model_vs_rule(self, iter_num: int=10):
        wins = 0
        turns = 0
        for _ in range(iter_num):
            winner, i = self.model_vs_rule()
            turns += i
            if winner == 'model':
                wins += 1
            if winner == 'draw':
                raise ValueError("Draw occurred, please check the model or the rule.")
        return wins / iter_num, turns / iter_num

class TestModel(BattleRandomAgent, BattleRuleAgent):

    NUM_OF_GAMES = 10

    def __init__(self, model_name='base_model', deck_name='', mode='random', iter_num=10):
        BattleRandomAgent.__init__(self, model_name, deck_name)
        BattleRuleAgent.__init__(self, model_name, deck_name)
        self.mode = mode
        self.iter_num = iter_num

    def test_base_model(self):
        model_info = {'win_rate': [], 'turns': []}
        with tqdm(total=self.iter_num) as pbar:
            for _ in range(self.iter_num):
                if self.mode == 'random':
                    win_info = self.calculate_win_rate_with_random(iter_num=self.NUM_OF_GAMES)
                elif self.mode == 'rule':
                    win_info = self.calculate_model_vs_rule(iter_num=self.NUM_OF_GAMES)
                else:
                    raise ValueError("mode must be 'random' or 'rule'")
                model_info['win_rate'].append(win_info[0])
                model_info['turns'].append(win_info[1])
                pbar.update(1)
        if self.deck != []:
            pd.to_pickle(model_info, 'pickle/' + self.model_name + f'_{self.mode}_{self.deck_name}' + '.pkl')
        else:
            pd.to_pickle(model_info, 'pickle/' + self.model_name + f'_{self.mode}' + '.pkl')

def check_model_info(model_name: str):
    model_info = pd.read_pickle('pickle/' + model_name + '.pkl')
    print(model_info)
    print('win rate:', np.mean(model_info['win_rate']))
    print('turns:', np.mean(model_info['turns']))
    print('std win rate:', np.std(model_info['win_rate'], ddof=1))
    print('std turns:', np.std(model_info['turns'], ddof=1))
    plt.plot(model_info['win_rate'])
    plt.show()

class MakeDeck:
    def __init__(self, model_name='make_deck_3000'):
        self.model_name = model_name
        self.model_path = 'models/' + self.model_name
        

    def test_make_deck(self):
        env = gym.make('MakeDeck-v0')
        model = DQN.load(self.model_path)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            if done:
                deck = info['deck']
                inserted_card = info['inserted_card']
                break
        pd.to_pickle(deck, 'decks/' + self.model_name + '.pkl')

if __name__ == '__main__':
    args = make_args()

    test_model_instance = TestModel(model_name=args.model_name, deck_name=args.deck_name, mode=args.mode, iter_num=args.iter_num)
    test_model_instance.test_base_model()