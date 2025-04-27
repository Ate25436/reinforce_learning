import argparse
import random
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from env import TCGEnv, TCGEnv_v2
from tools.stop_watch import stop_watch

def args():
    parser = argparse.ArgumentParser(description='DQN for TCG')
    parser.add_argument('--model_name', type=str, default='models/dqn_tcg', help='model name')
    parser.add_argument('--mode', type=str, default='log', help='log or calculate win rate')
    return parser.parse_args()

def battle_and_write(model_name):
    env = TCGEnv()
    model = DQN.load(model_name)
    random_agent = random.choice(['agent_0', 'agent_1'])


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
            if agent == random_agent:
                action = random.randrange(0, 40)
            else:
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
    

if __name__ == '__main__':
    args = args()
    model_name = args.model_name
    battle_and_write(model_name)
    