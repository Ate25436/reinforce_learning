import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import numpy as np
from numpy import random as rnd
import copy

MAX_STEP = 5000

class TCGEnv(ParallelEnv):
    metadata = {"render.mode": ["human"]}
    render_mode = 'human'
    MAX_TURN = 50
    CARD_ATTACK = 0
    CARD_HEALTH = 1
    CARD_PP = 2
    CARD_ABILITY = 3
    card_map = {
        'card_0': [4, 4, 1, 0],
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
        'card_14': [1, 1, 5, 2],
    }
    rewards_map = {
        'punish': -0.01,
        'reward': 0.0,
        'win': 1.0,
        'lose': -1.0
    }
    def __init__(self):
        self.agents = ['agent_0', 'agent_1']
        self.possible_agents = self.agents[:]
        self.TurnPlayer = 'agent_0'
        self.action_spaces = {'agent_0': Discrete(40), 'agent_1': Discrete(40)}
        self.observation_spaces = {'agent_0': Box(low=0, high=30, shape=(67, ), dtype=np.uint16), 'agent_1': Box(low=0, high=30, shape=(67, ), dtype=np.uint16)}
        self.turn = {'agent_0': 1, 'agent_1': 0}
        self.health = {'agent_0': 20, 'agent_1': 20}
        self.PP = {'agent_0': 1, 'agent_1': 0}
        self.hands = {'agent_0': [[0 for _ in range(4)] for _ in range(9)], 'agent_1': [[0 for _ in range(4)] for _ in range(9)]}
        self.fields = {'agent_0': [[0 for _ in range(2)] for _ in range(5)], 'agent_1': [[0 for _ in range(2)] for _ in range(5)]}
        self.attackable = {'agent_0': [0 for _ in range(5)], 'agent_1': [0 for _ in range(5)]}
        deck = []
        for i in range(15):
            deck += [self.card_map[f'card_{i}'] for _ in range(2)]
        self.decks = {'agent_0': copy.deepcopy(deck), 'agent_1': copy.deepcopy(deck)}
        self.trancated = {agent: False for agent in self.agents}
        self.steps = 0
    
    def create_observation(self):

        obs = {}
        for agent in self.agents:
            switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
            concated_obs = []
            concated_obs += [self.health[agent], self.health[switch_agent]]
            concated_obs += [self.PP[agent], self.PP[switch_agent]]
            concated_obs += flatten_list(self.hands[agent])
            concated_obs += flatten_list(self.fields[agent])
            concated_obs += flatten_list(self.fields[switch_agent])
            concated_obs += self.attackable[agent]
            concated_obs += [len(self.decks[agent]), len(self.decks[switch_agent])]
            obs[agent] = np.array(concated_obs).astype(np.uint16)
        return obs

    def step(self, actions):
        if all(self.turn[agent] >= self.MAX_TURN for agent in self.agents):
            return self.create_observation(), {agent: self.rewards_map['lose'] for agent in self.agents}, {agent: True for agent in self.agents}, self.trancated, {agent: {} for agent in self.agents}
        # if self.steps >= MAX_STEP:
        #     self.steps = 0
        #     return self.create_observation(), {agent: self.rewards_map['lose'] for agent in self.agents}, {agent: True for agent in self.agents}, self.trancated, {agent: {} for agent in self.agents}
        # self.steps += 1
        action = actions[self.TurnPlayer]
        agent = self.TurnPlayer
        if 0 <= action <= 8:
            obs, reward, done, info = self.play(agent, action)
            return obs, reward, done, self.trancated, info
        elif action == 39:
            obs, reward, done, info = self.end_turn(agent)
            return obs, reward, done, self.trancated, info
        else:
            obs, reward, done, info = self.attack(agent, (action - 9) // 6, (action - 9) % 6)
            return obs, reward, done, self.trancated, info

    def reset(self, seed=None, options=None):
        self.TurnPlayer = 'agent_0'
        self.turn = {'agent_0': 1, 'agent_1': 0}
        self.health = {'agent_0': 20, 'agent_1': 20}
        self.PP = {'agent_0': 1, 'agent_1': 0}
        self.hands = {'agent_0': [[0, 0, 0, 0] for _ in range(9)], 'agent_1': [[0, 0, 0, 0] for _ in range(9)]}
        self.fields = {'agent_0': [[0, 0] for _ in range(5)], 'agent_1': [[0, 0] for _ in range(5)]}
        self.attackable = {'agent_0': [0 for _ in range(5)], 'agent_1': [0 for _ in range(5)]}
        deck = []
        for i in range(15):
            deck += [self.card_map[f'card_{i}'] for _ in range(2)]
        self.decks = {'agent_0': copy.deepcopy(deck), 'agent_1': copy.deepcopy(deck)}
        self.draw_n('agent_0', 5)
        self.draw_n('agent_1', 5)
        return self.create_observation(), {agent: {} for agent in self.agents}


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
    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def play(self, agent, card_index):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        if self.hands[agent][card_index][self.CARD_HEALTH] == 0:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['punish'], switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}
        card_info = self.hands[agent][card_index]
        if card_info[self.CARD_PP] > self.PP[agent]:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['punish'], switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}
        field_index = self.find_empty_field(agent)
        if field_index == -1:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['punish'], switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}
        done = False
        self.PP[agent] -= card_info[self.CARD_PP]
        self.hands[agent][card_index] = [0, 0, 0, 0]
        self.fields[agent][field_index] = [card_info[self.CARD_ATTACK], card_info[self.CARD_HEALTH]]
        done, rewards = self.activate_ability(agent, card_info, field_index=field_index)
        return self.create_observation(), rewards, {agent: done, switch_agent: done}, {agent: {}, switch_agent: {}}
        
    def end_turn(self, agent):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        self.TurnPlayer = switch_agent
        self.turn[switch_agent] += 1
        self.PP[switch_agent] = min(self.turn[switch_agent], 8)
        done = self.draw_n(switch_agent, 1)
        if done:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['win'], switch_agent:self.rewards_map['lose']}, {agent: done, switch_agent: done}, {agent: {}, switch_agent: {}}
        for i in range(5):
            if self.fields[switch_agent][i][self.CARD_HEALTH] != 0:
                self.attackable[switch_agent][i] = 1
        observation = self.create_observation()
        return observation, {agent: 0.0, switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}

    def attack(self, agent, attacker_index, attacked_index):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        attacked_destruction = False
        attacker_destruction = False
        if self.fields[agent][attacker_index][self.CARD_HEALTH] == 0:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['punish'], switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}
        if attacked_index <= 4 and self.fields[switch_agent][attacked_index][self.CARD_HEALTH] == 0:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['punish'], switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}
        if self.attackable[agent][attacker_index] == 0:
            observation = self.create_observation()
            return observation, {agent: self.rewards_map['punish'], switch_agent:0.0}, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}
        if attacked_index <= 4:
            self.fields[switch_agent][attacked_index][self.CARD_HEALTH] -= self.fields[agent][attacker_index][self.CARD_ATTACK]
        elif attacked_index == 5:
            self.health[switch_agent] -= self.fields[agent][attacker_index][self.CARD_ATTACK]
        if attacked_index <= 4:
            self.fields[agent][attacker_index][self.CARD_HEALTH] -= self.fields[switch_agent][attacked_index][self.CARD_ATTACK]
        elif attacked_index == 5:
            pass

        if attacked_index <= 4 and self.fields[switch_agent][attacked_index][self.CARD_HEALTH] <= 0:
            self.fields[switch_agent][attacked_index] = [0, 0]
            self.attackable[switch_agent][attacked_index] = 0
            attacked_destruction = True
        if self.fields[agent][attacker_index][self.CARD_HEALTH] <= 0:
            self.fields[agent][attacker_index] = [0, 0]
            self.attackable[agent][attacker_index] = 0
            attacker_destruction = True
        self.attackable[agent][attacker_index] = 0
        if attacked_index <= 4:
            if attacked_destruction and attacker_destruction:
                rewards = {agent: self.rewards_map['reward'], switch_agent: 0.0}
            elif attacked_destruction:
                rewards = {agent: self.rewards_map['reward'], switch_agent: 0.0}
            elif attacker_destruction:
                rewards = {agent: 0.0, switch_agent: 0.0}
            else:
                rewards = {agent: self.rewards_map['reward'], switch_agent: 0.0}
        else:
            if self.health[switch_agent] <= 0:
                rewards = {agent: self.rewards_map['win'], switch_agent: self.rewards_map['lose']}
            else:
                rewards = {agent: self.rewards_map['reward'] * self.fields[agent][attacker_index][self.CARD_ATTACK], switch_agent: 0.0}
        observation = self.create_observation()
        return observation, rewards, {agent: False, switch_agent: False}, {agent: {}, switch_agent: {}}

    def activate_ability(self, agent, card_info, field_index=None):
        switch_agent = 'agent_0' if agent == 'agent_1' else 'agent_1'
        ability = card_info[self.CARD_ABILITY]
        mana_ratio = (card_info[self.CARD_ATTACK] + card_info[self.CARD_HEALTH]) / 2 * card_info[self.CARD_PP]
        match ability:
            case 0:   #能力なし
                return False, {agent: self.rewards_map['reward'] * mana_ratio, switch_agent: 0.00}
            case 1:   #召喚
                try:
                    i = self.fields[agent].index([0, 0])
                except ValueError:
                    return False, {agent: self.rewards_map['reward'] * mana_ratio, switch_agent: 0.00}
                self.fields[agent][i] = [1, 1]
                return False, {agent: self.rewards_map['reward'] * (1 + mana_ratio), switch_agent: 0.00}
            case 2:   #治癒
                old_health = self.health[agent]
                self.health[agent] = min(self.health[agent] + 2, 20)
                return False, {'agent_0': self.rewards_map['reward'] * mana_ratio + self.rewards_map['reward'] * (self.health[agent] - old_health), 'agent_1': 0.00}
            case 3:   #攻撃
                done = False
                self.health[switch_agent] -= 2
                if self.health[switch_agent] <= 0:
                    done = True
                    return done, {agent: self.rewards_map['win'], switch_agent: self.rewards_map['lose']}
                return done, {agent: self.rewards_map['reward'] * (2 + mana_ratio), switch_agent: 0.00}
            case 4:   #取得
                done = self.draw_n(agent, 1)
                if done:
                    return done, {agent: self.rewards_map['lose'], switch_agent: self.rewards_map['win']}
                return False, {agent: self.rewards_map['reward'] * (1 + mana_ratio), switch_agent: 0.00}
            case 5:   #速攻
                self.attackable[agent][field_index] = 1
                return False, {agent: self.rewards_map['reward'] * (1 + mana_ratio), switch_agent: 0.00}
            case _:  #その他
                return False, {agent: 0.0, switch_agent: 0.0}
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

    def switch_agent(self, agent):
        return 'agent_0' if agent == 'agent_1' else 'agent_1'
    
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
    def t_load_env(self, save_env):
        self.TurnPlayer = save_env['TurnPlayer']
        self.turn = copy.deepcopy(save_env['turn'])
        self.health = copy.deepcopy(save_env['health'])
        self.PP = copy.deepcopy(save_env['PP'])
        self.hands = copy.deepcopy(save_env['hands'])
        self.fields = copy.deepcopy(save_env['fields'])
        self.attackable = copy.deepcopy(save_env['attackable'])
        self.decks = copy.deepcopy(save_env['decks'])

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
    env = TCGEnv()
    obs, reward, done, info = env.step({'agent_0': 0, 'agent_1': 0})
    print(f"Step observation type: {type(obs['agent_0'])}")
if __name__ == '__main__':
    test()