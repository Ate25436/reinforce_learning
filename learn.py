import argparse

import gymnasium as gym
from stable_baselines3 import DQN

import custom_envs
from epsilon import ExponentialSchedule
from tools.stop_watch import stop_watch


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dqn_tcg")
    parser.add_argument("--timesteps", type=int, default=1000000)
    return parser.parse_args()

class CustomDQN(DQN):
    def __init__(self, *args, epsilon_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_schedule = epsilon_schedule or ExponentialSchedule()

    def _on_step(self) -> bool:
        self.exploration_rate = self.epsilon_schedule(self._n_updates)
        return True

# 5. SB3 で学習
@stop_watch
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
    learning_starts=10000,
    verbose=1,
    device="cuda"
)
    model.learn(total_timesteps=timesteps)
    model.save('models/' + model_name)
    return model

def main():
    # _, vec_env = make_vec_env()
    env = gym.make('MakeDeck-v0')
    timesteps = args().timesteps
    model_name = args().model_name
    model = learn_model(timesteps, env, model_name)

if __name__ == "__main__":
    main()

