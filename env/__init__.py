import gym

gym.register(
    id='CarRacing-v1',
    entry_point='env.car_racing:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)