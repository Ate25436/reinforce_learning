import matplotlib.pyplot as plt
import supersuit as ss
import torch
from pettingzoo.utils.conversions import parallel_to_aec
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import DQN

from env import TCGEnv

env = TCGEnv()
env = ss.flatten_v0(env)
gym_env = ss.pettingzoo_env_to_vec_env_v1(env)

vec_env = VecMonitor(gym_env)

model = DQN.load("turn_end_0")

obs, _ = vec_env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
print(f"Observation shape: {obs_tensor.shape}")
with torch.no_grad():
    q_values = model.q_net(obs_tensor)

# 可視化
q_values = q_values.cpu()
plt.bar(range(len(q_values[0])), q_values[0].numpy())
plt.xlabel("Actions")
plt.ylabel("Q-values")
plt.title("Q-values for each action")
plt.show()