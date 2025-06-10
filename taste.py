import argparse
import random
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from custom_envs.envs.field import TCGEnv
from tools.stop_watch import stop_watch

from tqdm import tqdm

def make_args():
    parser = argparse.ArgumentParser(description='DQN for TCG')
    parser.add_argument('--model_name', type=str, default='models/dqn_tcg', help='model name')
    parser.add_argument('--mode', type=str, default='battle', help='log or calculate win rate')
    return parser.parse_args()

def battle_and_write(model_name):                   # モデルに従うエージェントとランダムに行動するエージェントの戦いを記録する関数
                                                    # 記録はlog.txtに書き込まれる
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
        f.write(f'Battle information:\nrandom agent: {random_agent}, advance: {env.TurnPlayer}\n')
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
                f.write(f'TurnPlayer: {agent}\n')
                f.write(env.env_to_text() + '\n')
                # for key in save_env_before.keys():
                #     if save_env_before[key] != save_env_after[key] and key == 'TurnPlayer':
                #         f.write(key + '; ')
                #     elif save_env_before[key] != save_env_after[key]:
                #         change_agents = []
                #         for agent in env.agents:
                #             if save_env_before[key][agent] != save_env_after[key][agent]:
                #                 change_agents.append(agent)
                #         f.write(key + ': ' + ', '.join(change_agents)  + '; ')
                f.write('\n\n')
                i += 1
            done = terminated[agent]
            if done:
                winner = "agent_0" if rewards["agent_0"] == 1.0 else "agent_1"
        print(i)
        return winner
    

def battle_with_random(model_name):              # モデルに従うエージェントとランダムに行動するエージェントを戦わせる関数
    env = TCGEnv()
    model = DQN.load(model_name)
    random_agent = random.choice(['agent_0', 'agent_1'])
    
    obs, _ = env.reset()
    # env.render()
    done = False
    i = 0
    winner = ""
    while True:
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
            i += 1
        done = terminated[agent]
        if done:
            winner = "agent_0" if rewards["agent_0"] == 1.0 else("agent_1" if rewards["agent_1"] == 1.0 else "draw")
            break
    if winner == random_agent:
        winner = 'random'
    else:
        winner = 'model'
    return winner

def calculate_win_rate_with_random(model_name='dqn_tcg', iter_num=100):      # モデルに従うエージェントとランダムに行動するエージェントを戦わせて，勝率を計算する関数
    win = 0
    with tqdm(total=iter_num) as pbar:
        for _ in range(iter_num):
            winner = battle_with_random(model_name)
            if winner == 'model':
                win += 1
            pbar.set_postfix({'win': win})
            pbar.update(1)
    return win / iter_num

if __name__ == '__main__':
    args = make_args()
    model_name = 'models/' + args.model_name
    mode = args.mode
    if mode == 'calculate':
        iter_num = int(input('iter_num: '))
        win_rate = calculate_win_rate_with_random(model_name=model_name, iter_num=iter_num)
        print(f'Win rate: {win_rate}')
    elif mode == 'battle':
        battle_and_write(model_name)
    