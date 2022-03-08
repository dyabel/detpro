import torch
import sys

_, a, b = sys.argv
x, y = torch.load(a), torch.load(b)
x, y = x.squeeze(), y.squeeze()

x = x / x.norm(dim = -1, keepdim = True)
y = y / y.norm(dim = -1, keepdim = True)

x = x[:1203]
y = y[:1203]

sim = (x*y).sum(dim=-1)
print(sim)
print(sim.mean())