import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import numpy as np
from numpy import random as rnd
import copy
import random
from typing import List, Dict, Union
MAX_STEP = 5000

class TCGEnv_v2(gym.Env):
    metadata = {"render.modes": ["human"]}
    CARD_ATTACK = 0
    CARD_HEALTH = 1
    CARD_PP = 2
    CARD_ABILITY = 3
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
        'punish': -1e-2,
        'reward': 0.0,
        'win': 1.0,
        'lose': -1.0
    }

    def __init__(self):
        self.agents: List[str] = ['agent_0', 'agent_1']
        self.action_space = Discrete(40)
        self.observation_space = Box(low=0, high=30, shape=(67, ), dtype=np.uint16)
        self.LeadingPlayer: str = "agent_0" if random.randrange(2) == 0 else "agent_1"
        self.SecondPlayer: str = "agent_0" if self.LeadingPlayer == "agent_1" else "agent_1"
        self.turn = {self.LeadingPlayer: 1, self.SecondPlayer: 0}
        self.health = {self.LeadingPlayer: 20, self.SecondPlayer: 20}
        self.PP = {self.LeadingPlayer: 1, self.SecondPlayer: 0}
        self.hands = {self.LeadingPlayer: [[0 for _ in range(4)] for _ in range(9)], self.SecondPlayer: [[0 for _ in range(4)] for _ in range(9)]}
        self.fields = {self.LeadingPlayer: [[0 for _ in range(2)] for _ in range(5)], self.SecondPlayer: [[0 for _ in range(2)] for _ in range(5)]}
        self.attackable: Dict[str, List[int]] = {self.LeadingPlayer: [0 for _ in range(5)], self.SecondPlayer: [0 for _ in range(5)]}
        deck = []
        for i in range(15):
            deck += [self.card_map[f'card_{i}'] for _ in range(2)]
        self.decks = {"agent_0": copy.deepcopy(deck), "agent_1": copy.deepcopy(deck)}
        self.trancated = False
        self.agent_1_mode = "aggro" if random.randrange(2) == 0 else "control"
        self.ready = False
        self.inserted_card = {f"card_{i}": 0 for i in range(40)}
        
    def create_observation(self):
        concated_obs = []
        concated_obs += [self.health["agent_0"], self.health["agent_1"]]
        concated_obs += [self.PP["agent_0"], self.PP["agent_1"]]
        concated_obs += flatten_list(self.hands["agent_0"])
        concated_obs += flatten_list(self.fields["agent_0"])
        concated_obs += flatten_list(self.fields["agent_1"])
        concated_obs += self.attackable["agent_0"]
        concated_obs += [len(self.decks["agent_0"]), len(self.decks["agent_1"])]
        return np.array(concated_obs).astype(np.uint16)
        
    def step(self, action):
        # if self.ready:
        if 0 <= action <= 8:
            obs, reward, done, info = self.play("agent_0", action)
            return obs, reward, done, self.trancated, info
        elif action == 39:
            obs, reward, done, info = self.end_turn("agent_0")
            return obs, reward, done, self.trancated, info
        elif 9 <= action <= 38:
            obs, reward, done, info = self.attack("agent_0", (action - 9) // 6, (action - 9) % 6)
            return obs, reward, done, self.trancated, info
        else:
            return self.create_observation(), self.rewards_map['punish'], False, False, {}
        # else:
        #     if action < 40:
        #         return self.create_observation(), self.rewards_map['punish'], False, False, {}
        #     else:
        #         self.decks["agent_0"].append(self.card_map[f"card_{action - 40}"])
        #         self.inserted_card[f"card_{action - 40}"] += 1
        #         if len(self.decks["agent_0"]) == 30:
        #             self.ready = True
        #             self.draw_n("agent_0", 5)
        #             self.game_start()
        #         return self.create_observation(), 0.0, False, False, {}
            
    def game_start(self):
        if self.LeadingPlayer == "agent_1":
            self.agent_1_play(self.agent_1_mode)

    
    def reset(self, seed=None):
        self.LeadingPlayer = "agent_0"
        self.SecondPlayer = "agent_0" if self.LeadingPlayer == "agent_1" else "agent_1"
        self.TurnPlayer = self.LeadingPlayer
        self.turn = {self.LeadingPlayer: 1, self.SecondPlayer: 0}
        self.health = {self.LeadingPlayer: 20, self.SecondPlayer: 20}
        self.PP = {self.LeadingPlayer: 1, self.SecondPlayer: 0}
        self.hands = {self.LeadingPlayer: [[0, 0, 0, 0] for _ in range(9)], self.SecondPlayer: [[0, 0, 0, 0] for _ in range(9)]}
        self.fields = {self.LeadingPlayer: [[0, 0] for _ in range(5)], self.SecondPlayer: [[0, 0] for _ in range(5)]}
        self.attackable = {self.LeadingPlayer: [0 for _ in range(5)], self.SecondPlayer: [0 for _ in range(5)]}
        deck = []
        for i in range(15):
            deck += [self.card_map[f'card_{i}'] for _ in range(2)]
        self.decks = {"agent_0": [], "agent_1": copy.deepcopy(deck)}
        self.draw_n("agent_0", 5)
        self.draw_n("agent_1", 5)
        self.trancated = False
        self.agent_1_mode = "aggro" if random.randrange(2) == 0 else "control"
        self.ready = False
        self.inserted_card = {f"card_{i}": 0 for i in range(40)}
        self.game_start()
        return self.create_observation(), {}
    
    def render(self, mode='human'):
        if mode == 'human':
            for agent in self.agents:
                print(f'{agent}:')
                print(f'health: {self.health[agent]}, PP: {self.PP[agent]}')
                print(f'hand: {"; ".join(" ".join(str(item) for item in card) for card in self.hands[agent])}\n')
                print(f'field: {"; ".join(" ".join(str(item) for item in card) for card in self.fields[agent])}')
                print(f'attackable: {self.attackable[agent]}')
                print(f'deck_num: {len(self.decks[agent])}')
                print()

    def play(self, agent, card_index):
        if self.hands[agent][card_index][self.CARD_HEALTH] == 0:
            observation = self.create_observation()
            return observation, self.rewards_map['punish'], False, {}
        card_info = self.hands[agent][card_index]
        if card_info[self.CARD_PP] > self.PP[agent]:
            observation = self.create_observation()
            return observation, self.rewards_map['punish'], False, {}
        field_index = self.find_empty_field(agent)
        if field_index == -1:
            observation = self.create_observation()
            return observation, self.rewards_map['punish'], False, {}
        done = False
        self.PP[agent] -= card_info[self.CARD_PP]
        self.hands[agent][card_index] = [0, 0, 0, 0]
        self.fields[agent][field_index] = [card_info[self.CARD_ATTACK], card_info[self.CARD_HEALTH]]
        done, reward = self.activate_ability(agent, card_info, field_index=field_index)
        obs = self.create_observation()
        return obs, reward, done, {}
    
    def activate_ability(self, agent: str, card_info, field_index: Union[int, None]=None):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        ability = card_info[self.CARD_ABILITY]
        mana_ratio = (card_info[self.CARD_ATTACK] + card_info[self.CARD_HEALTH]) / 2 * card_info[self.CARD_PP]
        match ability:
            case 0:   #能力なし
                return False, self.rewards_map['reward'] * mana_ratio
            case 1:   #召喚
                try:
                    i = self.fields[agent].index([0, 0])
                except ValueError:
                    return False, self.rewards_map['reward'] * mana_ratio
                self.fields[agent][i] = [1, 1]
                return False, self.rewards_map['reward'] * (1 + mana_ratio)
            case 2:   #治癒
                old_health = self.health[agent]
                self.health[agent] = min(self.health[agent] + 2, 20)
                return False, self.rewards_map['reward'] * mana_ratio + self.rewards_map['reward'] * (self.health[agent] - old_health), 
            case 3:   #攻撃
                done = False
                self.health[switch_agent] -= 2
                if self.health[switch_agent] <= 0:
                    done = True
                    return done, self.rewards_map['win']
                return done, self.rewards_map['reward'] * (2 + mana_ratio), 
            case 4:   #取得
                done = self.draw_n(agent, 1)
                if done:
                    return done, self.rewards_map['lose']
                return False,  self.rewards_map['reward'] * (1 + mana_ratio)
            case 5:   #速攻
                if field_index is not None:
                    self.attackable[agent][field_index] = 1
                return False, self.rewards_map['reward'] * (1 + mana_ratio)
            case _:  #その他
                return False, 0.0

    def attack(self, agent, attacker_index, attacked_index):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        attacked_destruction = False
        attacker_destruction = False
        if self.fields[agent][attacker_index][self.CARD_HEALTH] == 0:  #エージェントが指定した自盤面にカードが存在しなかった場合
            observation = self.create_observation()
            return observation, self.rewards_map['punish'], False, {}
        if attacked_index <= 4 and self.fields[switch_agent][attacked_index][self.CARD_HEALTH] == 0:  #エージェントが指定した相手盤面にカードが存在しなかった場合
            observation = self.create_observation()
            return observation, self.rewards_map['punish'], False, {}
        if self.attackable[agent][attacker_index] == 0:  #エージェントが指定した自盤面のカードが攻撃可能でなかった場合
            observation = self.create_observation()
            return observation, self.rewards_map['punish'], False, {}
        if attacked_index <= 4:  #相手盤面のカードを攻撃するときの相手カードの処理
            self.fields[switch_agent][attacked_index][self.CARD_HEALTH] -= self.fields[agent][attacker_index][self.CARD_ATTACK]
        elif attacked_index == 5:  #相手プレイヤーを攻撃するときの処理
            self.health[switch_agent] -= self.fields[agent][attacker_index][self.CARD_ATTACK]
        if attacked_index <= 4:  #相手盤面のカードを攻撃するときの自カードの処理
            self.fields[agent][attacker_index][self.CARD_HEALTH] -= self.fields[switch_agent][attacked_index][self.CARD_ATTACK]
        elif attacked_index == 5:  #相手プレイヤーを攻撃するときの自カードの処理
            pass

        if attacked_index <= 4 and self.fields[switch_agent][attacked_index][self.CARD_HEALTH] <= 0:  #相手盤面のカードが破壊された場合
            self.fields[switch_agent][attacked_index] = [0, 0]
            self.attackable[switch_agent][attacked_index] = 0
            attacked_destruction = True
        if self.fields[agent][attacker_index][self.CARD_HEALTH] <= 0:  #自盤面のカードが破壊された場合
            self.fields[agent][attacker_index] = [0, 0]
            self.attackable[agent][attacker_index] = 0
            attacker_destruction = True
        self.attackable[agent][attacker_index] = 0
        done = False
        if attacked_index <= 4:
            if attacked_destruction and attacker_destruction:
                reward = self.rewards_map['reward']
            elif attacked_destruction:
                reward = self.rewards_map['reward']
            elif attacker_destruction:
                reward = 0.0
            else:
                reward = self.rewards_map['reward']
        else:
            if self.health[switch_agent] <= 0:
                reward = self.rewards_map['reward']
                done = True
            else:
                reward = self.rewards_map['reward'] * self.fields[agent][attacker_index][self.CARD_ATTACK]
        observation = self.create_observation()
        return observation, reward, done, {}
    
    def end_turn(self, agent):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        self.TurnPlayer = switch_agent
        self.turn[switch_agent] += 1
        self.PP[switch_agent] = min(self.turn[switch_agent], 8)
        done = self.draw_n(switch_agent, 1)
        if done:
            observation = self.create_observation()
            return observation, self.rewards_map['win'], done, {}
        for i in range(5):
            if self.fields[switch_agent][i][self.CARD_HEALTH] != 0:
                self.attackable[switch_agent][i] = 1
        done = False
        if switch_agent == "agent_1":
            done, reward = self.agent_1_play(self.agent_1_mode)
        observation = self.create_observation()
        if done:
            return observation, reward * -1, done, {}
        else:
            return observation, 0.0, done, {}

    def agent_1_play(self, mode):
        playable_cards_dict = {}
        for i in range(9):
            if self.hands["agent_1"][i][self.CARD_HEALTH] != 0 and self.hands["agent_1"][i][self.CARD_PP] <= self.PP["agent_1"]:
                manaratio = (self.hands["agent_1"][i][self.CARD_ATTACK] + self.hands["agent_1"][i][self.CARD_HEALTH]) / (2 * self.hands["agent_1"][i][self.CARD_PP])
                playable_cards_dict[i] = manaratio if self.hands['agent_1'][i][self.CARD_ABILITY] == 0 else manaratio + 1
        playable_cards = [k for k, v in sorted(playable_cards_dict.items(), key=lambda x: x[1], reverse=True)]
        for i in playable_cards:
            if self.find_empty_field("agent_1") != -1 and self.hands["agent_1"][i][self.CARD_PP] <= self.PP["agent_1"]:
                _, reward, done, _ = self.play("agent_1", i)
                if done:
                    return done, reward
        if mode == "aggro":
            for i in range(5):
                if self.attackable["agent_1"][i] == 1:
                    _, reward, done, _ = self.attack("agent_1", i, 5)
                    if done:
                        return done, reward
        elif mode == "control":
            for i in range(5):
                if self.attackable["agent_1"][i] == 1 and self.find_empty_field("agent_0") != -1:
                    attacked = False
                    for j in range(5):
                        if self.fields["agent_0"][j][self.CARD_HEALTH] != 0 and self.fields["agent_0"][j][self.CARD_ATTACK] < self.fields["agent_1"][i][self.CARD_HEALTH]:
                            self.attack("agent_1", i, j)
                            attacked = True
                            break
                    if not attacked:
                        _, reward, done, _ = self.attack("agent_1", i, 5)
                        if done:
                            return done, reward

        self.end_turn("agent_1")
        return False, 0.0
    
    def find_empty_field(self, agent):
        try:
            i = self.fields[agent].index([0, 0])
        except ValueError:
            return -1
        return i
    
    def find_empty_hand(self, agent):
        try:
            i = self.hands[agent].index([0, 0, 0, 0])
        except ValueError:
            return -1
        return i

    def draw_n(self, agent, n):
        for _ in range(n):
            deck = self.decks[agent]
            if len(deck) == 0:
                return True
            else:
                rnd.shuffle(deck)
                card = deck.pop()
                card_index = self.find_empty_hand(agent)
                if card_index != -1:
                    self.hands[agent][card_index] = card
        return False
    
    def env_to_text(self):
        text_list = [
            f'''
            agent: {agent}
            health: {self.health[agent]}, PP: {self.PP[agent]}
            hand: {"; ".join(" ".join(str(item) for item in card) for card in self.hands[agent])}
            field: {"; ".join(" ".join(str(item) for item in card) for card in self.fields[agent])}
            attackable: {self.attackable[agent]}
            deck_num: {len(self.decks[agent])}''' for agent in self.agents
        ]
        return '\n'.join(text_list)
    
    def t_save_env(self):
        save_env = {
            'TurnPlayer': self.TurnPlayer,
            'turn': copy.deepcopy(self.turn),
            'health': copy.deepcopy(self.health),
            'PP': copy.deepcopy(self.PP),
            'hands': copy.deepcopy(self.hands),
            'fields': copy.deepcopy(self.fields),
            'attackable': copy.deepcopy(self.attackable),
            'decks': copy.deepcopy(self.decks)
        }
        return save_env

def base_n(num_10,n):
    str_n = ''
    while num_10:
        if num_10%n>=10:
            return -1
        str_n += str(num_10%n)
        num_10 //= n
    return int(str_n[::-1])

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def test():
    env = TCGEnv_v2()
    env.reset()
    env.render()
    for i in range(30):
        env.step(i)
    env.render()


if __name__ == '__main__':
    test()