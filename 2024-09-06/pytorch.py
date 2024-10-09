import torch
import torch.nn as nn
import time
import torch.nn.functional as F

p = 100
steps = 1000

n = 1000

X = torch.randn(n, p, device = "cuda")
beta = torch.randn(p, 1, device = "cuda")
Y = X.matmul(beta) 

latent = 5000

net = nn.Sequential(
  nn.Linear(p, latent),
  nn.ReLU(),
  nn.Linear(latent, 1)
)

net.cuda()

net.compile()

opt = torch.optim.Adam(net.parameters())

t1 = time.time()
for i in range(steps):
    opt.zero_grad()
    Y_hat = net(X)
    loss = F.mse_loss(Y_hat, Y)
    x = loss.backward()
    opt.step()
t2 = time.time()

print(f"Time taken: {t2 - t1} seconds")

