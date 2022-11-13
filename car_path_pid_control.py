import gym
import torch
from env.car_path import params
from collections import deque


class PIDControlEvaluator:
    def __init__(self, num_runs=1, render=False, log=False, demo=False, fps=50, device='cpu', crash_penalty=0, max_episode_steps=500):
        self.num_runs = num_runs
        self.render, self.log, self.demo, self.fps = render, log, demo, fps
        self.device = device
        self.crash_penalty = crash_penalty
        self.max_episode_steps = max_episode_steps

    def __call__(self, PID, demo=False):
        N_CARS = PID.shape[0]
        env = gym.make('CarPath-v1', n_cars=N_CARS, max_episode_steps=self.max_episode_steps,
                       headless=not self.render, device=self.device, crash_penalty=self.crash_penalty)

        reward_total = torch.zeros(N_CARS, device=self.device)
        state_buffer = deque(maxlen=5)
        PID = PID.to(self.device)

        def policy(state, PID):
            with torch.no_grad():
                # buffer the last few states
                state_buffer.append(state)
                states = torch.stack(list(state_buffer), dim=-1)

                # find the center of the longest beam
                dist, index = torch.max(states, dim=1, keepdim=True)
                n_max = (states == dist).sum(dim=1, keepdim=True)
                index += n_max.div(2, rounding_mode='trunc')
                turn = (index - params['N_BEAMS'] // 2) / params['N_BEAMS'] * 2

                # compute turning angle using PID
                turn = turn.squeeze(1)
                d_turn = turn[:, -1] - turn[:, -2]
                P = turn[:, -1] * PID[:, 0]
                I = torch.minimum(turn.sum(1) * PID[:, 1], PID[:, 2])
                D = d_turn * PID[:, 3]
                turn = P + I + D

                # compute distance using PID
                dist = dist.squeeze(1)
                d_dist = dist[:, -1] - dist[:, -2]
                P = dist[:, -1] * PID[:, 4]
                I = torch.minimum(dist.sum(1) * PID[:, 5], PID[:, 6])
                D = d_dist * PID[:, 7]
                dist = P + I + D
                dist = dist.clamp(0.01, 1.)

                return torch.stack([turn, dist], dim=-1)

        for _ in range(self.num_runs):
            state = env.reset()
            state_buffer.append(state)
            env.render(fps=self.fps)
            done = torch.zeros(N_CARS, dtype=torch.bool)
            while not done.all():
                action = policy(state, PID)
                state, reward, done, info = env.step(action)
                reward_total += reward
                env.render(fps=self.fps)

        return reward_total


if __name__ == '__main__':
    N_CARS = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PID = torch.tensor([ 0.6617, -0.0042,  0.8482, -0.2855,  0.2150,  1.3173, -0.0842, -0.1718], device=device)
    PID = PID[None, :].repeat(N_CARS, 1)
    evaluate = PIDControlEvaluator(num_runs=2, render=True, fps=2, device=device, crash_penalty=1000.)
    print(evaluate(PID))
