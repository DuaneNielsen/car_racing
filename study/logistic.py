import torch
from torch import distributions
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.nn.functional import softplus
import matplotlib.pyplot as plt


def logistic(loc, scale):
    scale = softplus(scale) + torch.finfo(scale.dtype).eps
    lower, upper = torch.tensor([0.], device=loc.device), torch.tensor([1.], device=loc.device)
    base_distribution = distributions.Uniform(lower, upper)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
    return distributions.TransformedDistribution(base_distribution, transforms)


x = torch.linspace(0, 1, 100)
logprobs = logistic(loc=torch.tensor([0.5]), scale=torch.tensor([0.5])).log_prob(x)
plt.plot(x, logprobs.exp())

logprobs = logistic(loc=torch.tensor([0.5]), scale=torch.tensor([0.2])).log_prob(x)
plt.plot(x, logprobs.exp())

plt.show()