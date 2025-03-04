from gym.envs.registration import register

register(
    id='TCGEnv-v0',
    entry_point='env:TCGEnv_v2',
)