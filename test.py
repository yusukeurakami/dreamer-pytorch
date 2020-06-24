from utils import lambda_return, FreezeParameters
import torch
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import Transform, TanhTransform
from torch.nn import functional as F

transition_model = torch.nn.Linear(20, 10)
value_model = torch.nn.Linear(10, 2)

# with FreezeParameters([value_model]):
inp1 = torch.ones(20)
returns = torch.zeros(2)
inp2 = transition_model(inp1)
with torch.no_grad():
    inp2 = inp2.detach()
value_pred = value_model(inp2)
target_return = returns.detach()
transition_loss = F.mse_loss(value_pred, target_return).mean(dim=(0))

# with FreezeParameters([transition_model]):
#     inp1 = torch.ones(20)
#     returns = torch.zeros(2)
#     inp2 = transition_model(inp1)
#     value_pred = value_model(inp2)
#     target_return = returns.detach()
#     value_loss = F.mse_loss(value_pred, target_return).mean(dim=(0))

transition_loss.backward()
print(transition_model.weight.grad)
print(value_model.weight.grad)
# value_loss.backward()
# print(transition_model.weight.grad)
# print(value_model.weight.grad)


# disclam = 0.95
# discount = 0.99
# reward = torch.arange(0,30).view(3,2,5)
# flatten = lambda x: x.view([-1]+list(x.size()[2:]))
# value = torch.arange(30,60).view(3,10)

# print(flatten(reward))

# returns = lambda_return(
#     reward[:-1], value[:-1], bootstrap=value[-1], lambda_=disclam)

# print(returns)
# print(returns.size())

# PYTORCH
# [57.9398, 60.8608, 63.7819, 66.7030, 69.6241, 72.5452, 75.4663, 78.3874, 81.3085, 84.2296]
# [59.5000, 61.4900, 63.4800, 65.4700, 67.4600, 69.4500, 71.4400, 73.4300, 75.4200, 77.4100]

# TF
# [57.93975  60.860844 63.781944 66.70304  69.62414  72.54523  75.466324, 78.38741  81.30851  84.22961 ]
# [59.5   , 61.489998 63.480003 65.47     67.46001  69.45     71.44 73.43     75.42     77.41    ]]

# mean = torch.zeros((3,4))
# std = torch.ones((3,4))
# dist = Normal(mean, std)
# dist = TransformedDistribution(dist, TanhTransform())
# # sample = dist.sample_n(2)
# # print("sample: ", sample) # torch.Size([100, 1, 6])


# sample = torch.Tensor([[[ 0.5467947,  -0.1615259,   0.85917974,  0.8844119 ],
#                         [ 0.9430844,  -0.65128314, -0.37722188, -0.7573055 ],
#                         [ 0.97344655,  0.23397793, -0.5207381,  -0.8949507 ]],

#                         [[-0.02690708,  0.93916935, -0.7454547,   0.54994726],
#                         [-0.8515325,  -0.87329435,  0.5729679,  -0.7884654 ],
#                         [-0.4692143,  -0.7107713,  -0.4622952,   0.73246235]]])

# print(sample)

# logprob = dist.log_prob(sample)
# print("logprob: ",logprob)
# logprob = logprob.sum(2)
# print("logprob: ",logprob.size())
# print("logprob: ",logprob)
# # still not sure how the following parts work
# logprob_argmax = torch.argmax(logprob,0)
# # print("logprob argmax:", logprob_argmax) # torch.Size([1, 6])
# sample = sample[logprob_argmax]
# # print("sample selected: ",sample) #torch.Size([1, 6, 1, 6]) --> have to be [1, 6]
# print("return:", sample[0])