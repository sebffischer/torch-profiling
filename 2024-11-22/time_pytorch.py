import torch
import time
from torch import nn
import math

# 3. Define the timing function
def time_pytorch(epochs, batch_size, n_layers, latent, n, p, optimizer, device, num_threads, num_interop_threads, seed):
    return 1
    torch.manual_seed(seed)
    # 2. Define a function to create the neural network
    def make_network(p, latent, n_layers):
        layers = [nn.Linear(p, latent), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(latent, latent))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent, 1))
        return nn.Sequential(*layers)

    device = torch.device(device)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    
    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, 1, device=device)
    Y = X.matmul(beta) + torch.randn(n, 1, device=device) * 0.1
    
    # Create the network
    net = make_network(p, latent, n_layers)
    net.to(device)
    
    # Define optimizer and loss function
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)

    loss_fn = nn.MSELoss()


    def get_batch(step, X, Y, batch_size):
        start_index = (step - 1) * batch_size
        end_index = min(step * batch_size, X.size(0))  # Use X.size(0) to get the number of rows
        x_batch = X[start_index:end_index]
        y_batch = Y[start_index:end_index]
        return x_batch, y_batch

    steps = math.ceil(n / batch_size)

    def train_run():
        for _ in range(1, epochs + 1):
            for step in range(1, steps + 1):
                x, y = get_batch(step, X, Y, batch_size)
                optimizer.zero_grad()
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
            
        return loss.item()

    t0 = time.time()
    loss = train_run()
    t = time.time() - t0

    return 1
    
    #return {'time': t, 'loss': loss}

