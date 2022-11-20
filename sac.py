import torch
import torch.nn as nn
import torch.nn.functional as NNF
from torch.utils.data import DataLoader, RandomSampler

import gym
import env
import env.wrappers as wrappers
from gymviz import Plot
import numpy as np
import pybulletgym

from algos import sac
from distributions import ScaledTanhTransformedGaussian
from config import exists_and_not_none, ArgumentParser, EvalAction
import wandb
import wandb_utils
import checkpoint

if __name__ == '__main__':

    """ configuration """
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-r', '--run_id', type=int, default=-1)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--precision', type=str, action=EvalAction, default=torch.float32)

    """ reproducibility """
    parser.add_argument('--seed', type=int, default=None)

    """ main loop control """
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--test_sample_n_episodes', type=int, default=3)
    parser.add_argument('--test_max_steps', type=int, default=300)
    parser.add_argument('--test_every_n_steps', type=int, default=10000)
    parser.add_argument('--test_capture', action='store_true', default=False)
    parser.add_argument('--test_render', action='store_true', default=False)
    parser.add_argument('--test_fps', type=int, default=2)

    """ resume settings """
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('-l', '--load', type=str, default=None)

    """ environment """
    parser.add_argument('--env_name', type=str, default='CarPath-v1')
    parser.add_argument('--env_render', action='store_true', default=False)
    parser.add_argument('--env_reset_after_n_steps', type=int, default=300)
    parser.add_argument('--env_reward_scale', type=float, default=1.0)
    parser.add_argument('--env_reward_bias', type=float, default=0.0)

    """ hyper-parameters """
    parser.add_argument('--optim_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.005)
    parser.add_argument('--q_update_ratio', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--min_variance', type=float, default=0.01)

    config = parser.parse_args()

    """ random seed """
    if config.seed is not None:
        torch.manual_seed(config.seed)

    wandb.init(project=f"sac-{config.env_name}", config=config)

    """ environment """
    def make_env(n_cars, render_fps, max_episode_steps=None, headless=False):
        env = gym.make(config.env_name, n_cars=n_cars, render_fps=render_fps, headless=headless,
                       max_episode_steps=max_episode_steps, device=config.device
                       ).unwrapped
        if config.seed is not None:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
        return env


    """ training env with replay buffer """
    train_env = make_env(n_cars=config.batch_size, render_fps=100, headless=not config.env_render)
    if config.debug:
        train_env = Plot(train_env, episodes_per_point=5, title=f'Train sac-{config.env_name}')

    """ test env """
    test_env = make_env(n_cars=1, render_fps=config.test_fps, max_episode_steps=config.test_max_steps,
                        headless=not config.test_render)
    if config.debug:
        test_env = Plot(test_env, episodes_per_point=1, title=f'Test sac-{config.env_name}')


    class SoftMLP(nn.Module):
        def __init__(self, input_dims, hidden_dims, out_dims):
            super().__init__()
            self.hidden = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SELU(inplace=True),
                                        nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True))
            self.mu = nn.Linear(hidden_dims, out_dims)
            self.scale = nn.Linear(hidden_dims, out_dims)

        def forward(self, state):
            hidden = self.hidden(state)
            mu = self.mu(hidden)
            # scale = torch.sigmoid(self.scale(hidden)) + config.min_variance
            scale = NNF.softplus(self.scale(hidden)) + config.min_variance
            return mu, scale


    class Policy(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions, min_action, max_action):
            super().__init__()
            self.soft_mlp = SoftMLP(input_dims, hidden_dims, actions)
            self.min = min_action
            self.max = max_action
            # self.min = torch.tensor([-1., -1.])
            # self.max = torch.tensor([1., 1.])

        def forward(self, state):
            mu, scale = self.soft_mlp(state)
            return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


    class QNet(nn.Module):
        def __init__(self, input_dims, hidden_dims, actions, ensemble=2):
            super().__init__()
            self.q = [nn.Sequential(nn.Linear(input_dims + actions, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, hidden_dims), nn.SELU(inplace=True),
                                    nn.Linear(hidden_dims, 1)) for _ in range(ensemble)]

        def parameters(self, recurse=True):
            params = []
            for q in self.q:
                for param in q.parameters():
                    params.append(param)
            return params

        def forward(self, state, action):
            sa = torch.cat((state, action), dim=1)
            values = []
            for q in self.q:
                values += [q(sa)]
            return torch.stack(values, dim=-1)

        def to(self, device):
            self.q = [q.to(device) for q in self.q]
            return self

    q_net = QNet(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.shape[0],
        hidden_dims=config.hidden_dim).to(config.device)

    target_q_net = QNet(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.shape[0],
        hidden_dims=config.hidden_dim).to(config.device)

    assert np.all(test_env.action_space.low == test_env.action_space.low[0]), "action spaces do not have the same min"
    assert np.all(test_env.action_space.high == test_env.action_space.high[0]), "action spaces do not have the same max"
    assert len(test_env.observation_space.shape) == 1, "only 1-D observation spaces are supported"

    policy_net = Policy(
        input_dims=test_env.observation_space.shape[0],
        actions=test_env.action_space.shape[0],
        hidden_dims=config.hidden_dim,
        min_action=test_env.action_space.low[0].item(),
        max_action=test_env.action_space.high[0].item()
    ).to(config.device)

    q_optim = torch.optim.Adam(q_net.parameters(), lr=config.optim_lr)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=config.optim_lr)

    """ load weights from file if required"""
    if exists_and_not_none(config, 'load'):
        checkpoint.load(config.load, prefix='best', q=q_net, q_optim=q_optim, policy=policy_net,
                        policy_optim=policy_optim)

    """ policy to run on environment """


    def policy(state):
        with torch.no_grad():
            action = policy_net(state)
            a = action.rsample()
            assert ~torch.isnan(a).any()
            return a


    """ policy to run on environment """


    def exploit_policy(state):
        with torch.no_grad():
            action = policy_net(state)
            a = action.mean
            assert ~torch.isnan(a).any()
            return a


    """ demo  """
    wandb_utils.demo(config.demo, test_env, policy)

    """ train loop """
    evaluator = wandb_utils.Evaluator()
    buffer = wandb_utils.VectorStateBufferDataset(maxlen=100000)
    dl = None
    render_time = 0

    for step, (s, a, s_p, r, d, i) in enumerate(wandb_utils.step_environment(train_env, policy, buffer,
                                                                             reset_after_n_steps=config.env_reset_after_n_steps,
                                                                             render=config.env_render)):
        buffer.append((s, a, s_p, r, d), i)

        if dl is None:
            dl = DataLoader(buffer, batch_size=config.batch_size, sampler=RandomSampler(buffer, replacement=True))

        # give pygame a bit of time so that linux doensnt complain
        if config.test_render and not config.env_render and render_time % 100 == 0:
            # and why notrender a progress bar ?
            progress = ((step % config.test_every_n_steps) * 10 // config.test_every_n_steps)
            test_env.render(text=['[' + '#'*progress + '  ' * (10-progress) + ']'])
        render_time += 1

        if len(buffer) < config.batch_size * config.q_update_ratio:
            continue

        """ train online after batch steps saved"""
        sac.train(dl, q_net, target_q_net, policy_net, q_optim, policy_optim,
                  discount=config.discount, polyak=config.polyak, q_update_ratio=config.q_update_ratio,
                  alpha=config.alpha, device=config.device, precision=config.precision)

        """ test """
        if evaluator.evaluate_now(config.test_every_n_steps):
            evaluator.evaluate(test_env, exploit_policy, run_dir=config.run_dir, capture=config.test_capture,
                               render=config.test_render, sample_n=config.test_sample_n_episodes,
                               params={'q': q_net, 'q_optim': q_optim, 'policy': policy_net,
                                       'policy_optim': policy_optim})
            reset_train_env = True

        if step > config.max_steps:
            break
