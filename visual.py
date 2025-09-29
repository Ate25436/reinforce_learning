import json
from PIL import Image, ImageDraw, ImageFont
import tqdm

# 背景やアセット
BACKGROUND = "card_visual/board.png"
CARD_BACK = "card_visual/card_back.png"
FONT_PATH = "card_visual/Roboto/DroidSans-Bold.ttf"
ATTACK_ICON = "card_visual/attack_icon.png"
HEALTH_ICON = "card_visual/health_icon.png"
COST_ICON = "card_visual/cost_icon.png"
FRAME = "card_visual/frame.png"

with open('json/card_data.json', 'r') as f:
    CARD_DATA = json.load(f)


def draw_card(base, card_name, pos, place="hand", info=[]):
    """カード1枚を描画"""
    art = Image.open(CARD_DATA[card_name]["img"]).convert("RGBA")
    art = art.resize((150, 200))  # サイズ調整
    frame = Image.open(FRAME).convert("RGBA").resize((150, 200))
    base.paste(frame, pos, frame)
    base.paste(art, pos, art)

    draw = ImageDraw.Draw(base)
    font = ImageFont.truetype(FONT_PATH, 24)

    # 攻撃力・体力
    if place == "hand":
        attack = CARD_DATA[card_name]["attack"]
        health = CARD_DATA[card_name]["health"]
    elif place == "field":
        attack = info[0]
        health = info[1]
    attack_icon = Image.open(ATTACK_ICON).convert("RGBA").resize((30, 30))
    health_icon = Image.open(HEALTH_ICON).convert("RGBA").resize((30, 30))
    base.paste(attack_icon, (pos[0], pos[1]+170), attack_icon)
    base.paste(health_icon, (pos[0]+110, pos[1]+170), health_icon)
    draw.text((pos[0]+10, pos[1]+170), str(attack), font=font, fill="white")
    draw.text((pos[0]+120, pos[1]+170), str(health), font=font, fill="white")
    if place == "hand":
        # コスト
        cost = CARD_DATA[card_name]["cost"]
        cost_icon = Image.open(COST_ICON).convert("RGBA").resize((30, 30))
        base.paste(cost_icon, (pos[0], pos[1]), cost_icon)
        draw.text((pos[0]+10, pos[1]), str(cost), font=font, fill="white")

def render_game(state_json, output_folder="card_visual/boards"):
    with open(state_json, "r", encoding="utf-8") as f:
        states = json.load(f)
    with tqdm.tqdm(total=len(states)) as pbar:
        for step, state in enumerate(states):
            base = Image.open(BACKGROUND).convert("RGBA")
            draw = ImageDraw.Draw(base)
            font = ImageFont.truetype(FONT_PATH, 32)

            # 相手の手札（裏面）
            for i, card_name in enumerate(state["agent_1"]["hand"]):
                draw_card(base, card_name, (100 + i*150, 50), place="hand")

            # 相手の場
            for i, card_info in enumerate(state["agent_1"]["field"]):
                if len(card_info) == 2:
                    continue
                draw_card(base, card_info[-1], (200 + i*160, 250), place="field", info=card_info[0:2])

            # 自分の場
            for i, card_info in enumerate(state["agent_0"]["field"]):
                if len(card_info) == 2:
                    continue
                draw_card(base, card_info[-1], (200 + i*160, 500), place="field", info=card_info[0:2])

            # 自分の手札
            for i, card_name in enumerate(state["agent_0"]["hand"]):
                draw_card(base, card_name, (100 + i*150, 750), place="hand")

            # ライフ表示
            draw.text((50, 500), f"Life: {state['agent_0']['health']}", font=font, fill="green")
            draw.text((50, 250), f"Life: {state['agent_1']['health']}", font=font, fill="green")
            draw.text((50, 450), f"PP: {state['agent_0']['PP']}", font=font, fill="blue")
            draw.text((50, 300), f"PP: {state['agent_1']['PP']}", font=font, fill="blue")

            # 保存
            base.save(f"{output_folder}/step_{step}.png")
            pbar.update(1)

if __name__ == "__main__":
    render_game("json/battle_log.json", "card_visual/boards")
