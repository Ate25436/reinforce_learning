import json

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

card_dict = {}
for k, v in card_map.items():
    card_info = {
        "attack": v[0],
        "health": v[1],
        "cost": v[2],
        "ability": v[3],
        "img": f"card_visual/card/{k}.png"
    }
    card_dict[k] = card_info

with open('json/card_data.json', 'w') as f:
    json.dump(card_dict, f, indent=4)