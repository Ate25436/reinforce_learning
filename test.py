import argparse

from env import TCGEnv
from stable_baselines3 import DQN
from tools.stop_watch import stop_watch

def args():
    parser = argparse.ArgumentParser(description='DQN for TCG')
    parser.add_argument('--model_name', type=str, default='dqn_tcg', help='model name')
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
    return winner

@stop_watch
def calculate_win_rate(model_name='dqn_tcg', iter_num=100):
    win = 0
    for _ in range(iter_num):
        winner = battle(model_name)
        if winner == 'agent_0':
            win += 1
    return win / iter_num

def check_model(model_name):
    env = TCGEnv()
    model = DQN.load(model_name)

    obs, _ = env.reset()
    obs, rewards, terminated, _, _ = env.step({'agent_0': 6, 'agent_1': 0})
    print(rewards)

if __name__ == '__main__':
    # battle_and_write('dqn_tcg')
    args = args()
    model_name = args.model_name
    check_model(model_name)