from gymnasium.envs.registration import register

register(
    id='gymnasium-v0',
    entry_point='gymnasium.envs:Environment',
)