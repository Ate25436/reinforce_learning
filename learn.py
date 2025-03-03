import supersuit as ss
from pettingzoo.utils.conversions import parallel_to_aec
from pettingzoo.utils.wrappers import BaseWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from gymnasium.wrappers import TimeLimit

from env import TCGEnv


timesteps = 1000000

def make_vec_env():
    env = TCGEnv()
    gym_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = VecMonitor(gym_env)
    return gym_env, vec_env

# 5. SB3 で学習
def learn_model(timesteps, vec_env, model_name):
    model = DQN("MlpPolicy",
    vec_env,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    learning_starts=10000,
    verbose=1,
    device="cuda"
)
    model.learn(total_timesteps=timesteps)
    model.save(model_name)
    return model

def main():
    _, vec_env = make_vec_env()
    model = learn_model(timesteps, vec_env, "dqn_tcg")

if __name__ == "__main__":
    main()

