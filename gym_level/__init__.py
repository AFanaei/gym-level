from gym.envs.registration import register

register(
    id='level-v0',
    entry_point='gym_level.envs:LevelEnv',
    max_episode_steps=50,
)
register(
    id='jacket-v0',
    entry_point='gym_level.envs:JacketEnv',
    max_episode_steps=50,
)
