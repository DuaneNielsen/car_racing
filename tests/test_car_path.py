import gym
import env


def test_make():
    env = gym.make('CarPath-v1')
    obs = env.reset()
    env.render()
    env.close()