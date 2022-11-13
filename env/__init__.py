import gym

gym.register(
    id='CarRacing-v1',
    entry_point='env.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

gym.register(id="racer-v0", entry_point="env.gym_racer:RacerEnv")

gym.register(
    id='CarPath-v1',
    entry_point='env.car_path:CarRacingPathEnv',
    max_episode_steps=1000,
    reward_threshold=900,
)
