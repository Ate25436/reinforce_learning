from gymnasium.envs.registration import register

register(
    id='TCGEnv-v0', 
    entry_point='custom_envs.envs.field:TCGEnv',
)

register(
    id='TCGEnv-v2',
    entry_point='custom_envs.envs.tcg_env:TCGEnv_v2',
)

register(
    id='MakeDeck-v0',
    entry_point='custom_envs.envs.make_deck:MakeDeck',
)
