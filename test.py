import argparse
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st
from stable_baselines3 import DQN
from tqdm import tqdm

from env import TCGEnv_v2
from field import TCGEnv
from tools.stop_watch import stop_watch


def make_args():
    parser = argparse.ArgumentParser(description='DQN for TCG')
    parser.add_argument('--model_name', type=str, default='dqn_tcg', help='model name')
    parser.add_argument('--iter_num', type=int, default=10, help='number of iterations')
    return parser.parse_args()

def battle_and_write(model_name):
    env = TCGEnv()
    model = DQN.load(model_name)

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

def battle(model_name):
    action_list = []
    env = TCGEnv()
    model = DQN.load(model_name)

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

def battle_memory_actions(model_name):
    action_list = []
    random_list = []
    env = TCGEnv()
    model = DQN.load(model_name)

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
        random_list.append(random.randrange(0, 40))
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
    return winner, action_list, random_list

def battle_with_random(model_name):
    env = TCGEnv()
    model = DQN.load(model_name)
    random_agent = random.choice(['agent_0', 'agent_1'])
    
    obs, _ = env.reset()
    env.render()
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

@stop_watch
def calculate_win_rate(model_name='dqn_tcg', iter_num=100):
    win = 0
    for _ in range(iter_num):
        winner, _, _ = battle_memory_actions(model_name)
        if winner == 'agent_0':
            win += 1
    return win / iter_num

def calculate_win_rate_with_random(model_name='dqn_tcg', iter_num=100):
    win = 0
    turn_sum = 0
    for _ in range(iter_num):
        winner, i = battle_with_random(model_name)
        turn_sum += i
        if winner == 'model':
            win += 1
        elif winner == 'draw':
            print('here')
    return win / iter_num, turn_sum / iter_num

def comparison_action(model_name, iter_num=100):
    all_actions = []
    all_randoms = []
    for _ in range(10):
        action_list, random_list = battle(model_name)
        all_actions.extend(action_list)
        all_randoms.extend(random_list)
    actions_counter = Counter(all_actions)
    randoms_counter = Counter(all_randoms)
    fig, ax = plt.subplots(1, 2)
    ax[0].bar(actions_counter.keys(), actions_counter.values())
    ax[1].bar(randoms_counter.keys(), randoms_counter.values())
    plt.savefig('actions.png')
    plt.show()
    
def test_deck_make(model_name):
    env = TCGEnv_v2()
    obs, _ = env.reset()
    model = DQN.load(model_name)
    action_list = []
    while env.ready == False:
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
        if action >= 40:
            action_list.append(int(action))
    print(env.decks['agent_0'])
    return action_list

def test_base_model(iter_num=10):
    model_name_1 = 'models/potential_base_2'
    model_name_2 = 'models/potential_base_2_p'
    win_rates = {'model_1': [], 'model_2': []}
    with tqdm(total=iter_num) as pbar:
        for _ in range(iter_num):
            win_info_1 = calculate_win_rate_with_random(model_name_1, iter_num=10)
            win_info_2 = calculate_win_rate_with_random(model_name_2, iter_num=10)
            win_rates['model_1'].append(win_info_1)
            win_rates['model_2'].append(win_info_2)
            pbar.update(1)
    pd.to_pickle(win_rates, 'pickle/win_rates.pkl')

def test_base_model_turn(iter_num=30):
    model_name_1 = 'models/potential_base_2'
    model_name_2 = 'models/potential_base_2_p'
    win_rates = {'model_1': [], 'model_2': []}
    for _ in range(iter_num):
        _, turns = battle_with_random(model_name_1)
        win_rates['model_1'].append(turns)
        _, turns = battle_with_random(model_name_2)
        win_rates['model_2'].append(turns)
    pd.to_pickle(win_rates, 'pickle/turns.pkl')

def check_win_rate(pickle_name):
    win_rates_pickle = pd.read_pickle(pickle_name)
    win_rates = {key: [int(tuple[0] * 10) for tuple in value] for key, value in win_rates_pickle.items()}
    turn_mean = {key: [tuple[1] for tuple in value] for key, value in win_rates_pickle.items()}
    print(win_rates)
    print('win rates:', *[f"{key}: {np.mean(value)}" for key, value in win_rates.items()])
    print('turn mean:', *[f"{key}: {np.mean(value)}" for key, value in turn_mean.items()])
    
    x_1 = pd.Series(win_rates['model_1'])
    x_2 = pd.Series(win_rates['model_2'])
    x_label = list(range(11))
    x_1_value_count = x_1.value_counts()

    y_label = [x_1_value_count.get(i, 0) for i in x_label]
    plt.bar(x_label, y_label)
    plt.show()
    _, p_1 = st.shapiro(x_1)
    _, p_2 = st.shapiro(x_2)
    print(f'p_1: {p_1:.3f}')
    print(f'p_2: {p_2:.3f}')
    print(f'var_1: {x_1.std(ddof=1):.3f}')
    print(f'var_2: {x_2.std(ddof=1):.3f}')

    _, lev = st.levene(x_1, x_2, center='mean')
    print(f'levene p: {lev:.3f}')

    _, p_l = st.ttest_ind(x_1, x_2, equal_var=True)
    print(f'ttest p: {p_l}')
    _, p_w = st.ttest_ind(x_1, x_2, equal_var=False)
    print(f'welch ttest p: {p_w}')

def check_turns(pickle_name: str):
    turns_pickle = pd.read_pickle(pickle_name)
    x_1 = pd.Series(turns_pickle['model_1'])
    x_2 = pd.Series(turns_pickle['model_2'])
    print(turns_pickle)
    print('turns:', *[f"{key}: {np.mean(value)}" for key, value in turns_pickle.items()])
    _, p_1 = st.shapiro(x_1)
    _, p_2 = st.shapiro(x_2)
    print(f'p_1: {p_1:.3f}')
    print(f'p_2: {p_2:.3f}')

if __name__ == '__main__':
    # battle_and_write('dqn_tcg')
    args = make_args()
    check_win_rate('pickle/win_rates.pkl')