from gym.envs.registration import register

register(
    id='TCGEnv-v0',
    entry_point='envs.tcg_env:TCGEnv_v2',
)

register(
    id='MakeDeck-v0',
    entry_point='envs.make_deck:MakeDeck',
)