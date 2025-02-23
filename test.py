from env import TCGEnv

from stable_baselines3 import DQN



iter_num = 100
def battle_and_write(model_name):
    env = TCGEnv()
    model = DQN.load(model_name)

    obs, _ = env.reset()
    env.render()
    done = False
    i = 0
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
        print(i)

def battle(model_name):
    env = TCGEnv()
    model = DQN.load(model_name)

    obs, _ = env.reset()
    env.render()
    done = False
    i = 0
    while not done:
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
    print(i)

if __name__ == '__main__':
    battle_and_write('dqn_tcg')
    # battle('dqn_tcg')