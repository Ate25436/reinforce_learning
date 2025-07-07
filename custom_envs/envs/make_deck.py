import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from custom_envs.envs.utils import flatten_list
from custom_envs.envs.tcg_env import TCGEnv_v2

class MakeDeck(gym.Env):
    metadata = {"render.modes": ["human"]}
    card_map = {
        'card_0': [4, 4, 1, 0], # あえて強くした
        'card_1': [2, 2, 2, 0],
        'card_2': [3, 3, 3, 0],
        'card_3': [4, 3, 4, 0],
        'card_4': [5, 4, 5, 0],
        'card_5': [2, 2, 2, 1],
        'card_6': [2, 3, 3, 1],
        'card_7': [1, 1, 1, 4],
        'card_8': [1, 3, 2, 4],
        'card_9': [2, 1, 2, 5],
        'card_10': [3, 1, 3, 5],
        'card_11': [1, 2, 2, 3],
        'card_12': [2, 3, 3, 3],
        'card_13': [1, 1, 1, 2],
        'card_14': [1, 1, 5, 2], # あえて弱くした
        'card_15': [1, 4, 3, 2],
        'card_16': [2, 1, 2, 2],
        'card_17': [2, 2, 2, 2],
        'card_18': [3, 2, 3, 2],
        'card_19': [2, 5, 4, 2],
        'card_20': [5, 2, 4, 5], # まあまあ強い
        'card_21': [7, 1, 4, 0],
        'card_22': [3, 3, 4, 5],
        'card_23': [3, 1, 2, 3],
        'card_24': [1, 3, 2, 2],
        'card_25': [2, 2, 3, 2],
        'card_26': [2, 6, 5, 2],
        'card_27': [1, 2, 1, 0],
        'card_28': [2, 1, 1, 0],
        'card_29': [1, 1, 1, 1],
        'card_30': [1, 1, 1, 3],
        'card_31': [3, 2, 3, 1],
        'card_32': [4, 2, 4, 3],
        'card_33': [7, 2, 5, 3],
        'card_34': [1, 1, 1, 5],
        'card_35': [2, 7, 5, 1],
        'card_36': [2, 6, 5, 4],
        'card_37': [3, 7, 5, 0],
        'card_38': [5, 3, 4, 0],
        'card_39': [4, 3, 4, 2],
    }
    rewards_map = {
        'punish': 0.0,
        'reward': 0.0,
        'win': 1.0,
        'lose': -1.0
    }

    def __init__(self, model='base_model'):
        self.action_space = Discrete(10)
        self.observation_space = Box(low=0, high=10, shape=(120, ), dtype=np.uint16)
        self.deck = [[0, 0, 0, 0] for _ in range(30)]
        self.inserted_card = {f"card_{i}": 0 for i in range(40)}
        self.num_cards = 0
        self.model = DQN.load('models/' + model)
        self.id_list = list()

    def reset(self, seed=None, options=None):
        self.deck = [[0, 0, 0, 0] for _ in range(30)]
        self.inserted_card = {f"card_{i}": 0 for i in range(40)}
        self.id_list = list()
        self.num_cards = 0
        return self.create_observation(), {}
    
    def create_observation(self):
        concated_obs = []
        concated_obs += flatten_list(self.deck)
        return np.array(concated_obs).astype(np.uint16)
    
    def step(self, action):
        self.deck[self.num_cards] = self.card_map[f"card_{action}"]
        self.inserted_card[f"card_{action}"] += 1
        self.id_list.append(f"card_{action}")
        self.num_cards += 1
        if self.num_cards == 30:
            winner = self.battle()
            if winner == "model":
                reward = self.rewards_map['win']
            elif winner == "rule":
                reward = self.rewards_map['lose']
            return self.create_observation(), reward, True, False, {"deck": self.deck, "inserted_card": self.inserted_card}
        else:
            return self.create_observation(), 0.0, False, False, {}
    
    def render(self, mode='human'):
        top_5_cards = sorted(self.inserted_card.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5: ")
        for card, count in top_5_cards:
            print(f"{card}: {count}")
        print()
        print(*self.id_list)
        print(*self.deck)

    def battle(self):
        env = TCGEnv_v2()
        obs, _ = env.reset(deck=self.deck)
        done = False
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                if reward == 1.0:
                    winner = "model"
                elif reward == -1.0:
                    winner = "rule"
                break
        return winner
    
    def get_card_map(self):
        return self.card_map
    
    def get_deck(self):
        return self.deck

