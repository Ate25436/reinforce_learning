# カードゲームの強化学習

こちらの論文を参考に作成: https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2M5GS1002/_pdf/-char/ja

ボード情報（論文より引用，拡張性を考慮して一部変更）

|項目|次元数|上限|下限|
|---|---|---|---|
|各プレイヤーの体力|2|20|0|
|各プレイヤーのマナ|2|8|0|
|手札の攻撃力，HP，コスト，体力|36|8|0|
|自盤面の攻撃力，HP|10|8|0|
|相手盤面の攻撃力，HP|10|8|0|
|自盤面が攻撃可能か|5|1|0|
|各プレイヤーのデッキ枚数|2|30|0|

各エージェントが自分のターン中に取れる動き

- 手札9枚のうち1枚をプレイ（コストを支払って場に出す，カードが能力を持っている場合は能力発動）
- 自盤面5枚のうち1枚で相手盤面のカードor相手プレーヤーを攻撃
- ターンエンド

学習方法はDQN(Deep Q-Learning)
## agent_1の行動指針

- play
  1. プレイできるカードを，マナレシオが高い順に並べる
  2. マナレシオが高い順に，プレイできるならプレイする
- attack
  - aggro
    - とりあえず顔を殴る
  - control
    - 相手の場にこちらが不利トレードしなくて済む（相手の攻撃力より自分の体力のほうが高い場合）はそのカードに攻撃
    - いない場合は相手の顔を殴る

## 各ファイルの説明

- env.py 
  - 強化学習の環境が書かれたファイル
  - PettingZooのParallelEnvで書かれている
- learn.py
  - 強化学習を行うファイル
  - DQNで学習する（ハイパラはファイル参照）
- eval.py
  - 学習したモデルのq値を見るためのファイル
  - あまり使っていない
- test.py
  - 学習したモデルで推論のテストを行うファイル
  - エージェントの行動はlog.txtに出力される
- memo.md
  - 強化学習の報酬が記載されたファイル
  - 今後他の内容も乗せる可能性あり
## 用いているライブラリ
- gymnasium
- pettingzoo
- stable-baseline3
- pytorch
- supersuit
- isort

## 追記
- 2025/02/24
  - 不能な操作に対してバツを与える環境とそうでない環境で勝率・対戦時間の変化を見た．
  - なぜか，バツを与える環境のほうが対戦時間が長くなった．