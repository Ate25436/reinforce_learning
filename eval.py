import matplotlib.pyplot as plt
import supersuit as ss
import torch
from pettingzoo.utils.conversions import parallel_to_aec
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import DQN

from env import TCGEnv, TCGEnv_v2

env = TCGEnv_v2()

model = DQN.load("models/deck_10_with_punish_2")

obs, _ = env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
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