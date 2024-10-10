import torch
import itertools
import time
import os
import pickle
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd
import math

test = False

# 1. Define the configuration grid
if test:
    config_grid = {
        'n': [100],
        'p': [10, 20],
        'epochs': [2],
        'batch_size': [16, 32],
        'device': ['cpu', 'cuda'],
        'jit': [True, False],
        'latent': [10, 50, 100],
        'n_layers': [1, 5]
    }
else:
    config_grid = {
        'n': [2000],
        'p': [1000],
        'optimizer': ['adam', 'sgd'],
        'epochs': [10],
        'batch_size': [1024, 128, 16],
        'device': ['cuda'],
        'jit': [True, False],
        'latent': [2000, 1000, 500],
        'n_layers': [16, 4, 1]
    }

# Create all combinations of configurations
keys, values = zip(*config_grid.items())
configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

# 2. Define a function to create the neural network
def make_network(p, latent, n_layers):
    layers = [nn.Linear(p, latent), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(latent, latent))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(latent, 1))
    return nn.Sequential(*layers)

# 3. Define the timing function
def time_config(config):
    # set cuda seed to 123
    torch.manual_seed(123)
    # Select device
    device = torch.device(config['device'])
    
    # Generate random data
    X = torch.randn(config['n'], config['p'], device=device)
    beta = torch.randn(config['p'], 1, device=device)
    Y = X.matmul(beta) + torch.randn(config['n'], 1, device=device) * 0.1
    
    # Create the network
    net = make_network(config['p'], config['latent'], config['n_layers'])
    net.to(device)
    
    # Optionally apply JIT compilation
    if config['jit']:
        example_input = X[:config['batch_size']]
        net = torch.jit.trace(net, example_input)
    
    
    # Define optimizer and loss function
    if config['optimizer'] == 'adam':
        optimizer = Adam(net.parameters())
    else:
        optimizer = SGD(net.parameters(), lr = 0.001)

    loss_fn = nn.MSELoss()


    def get_batch(step, X, Y, batch_size):
        start_index = (step - 1) * batch_size
        end_index = min(step * batch_size, X.size(0))  # Use X.size(0) to get the number of rows
        x_batch = X[start_index:end_index]
        y_batch = Y[start_index:end_index]
        return {'x': x_batch, 'y': y_batch}

    
    # Record start time
    start_time = time.time()

    steps = math.ceil(config['n'] / config['batch_size'])


    timings = []
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        for step in range(1, steps + 1):
            batch = get_batch(step, X, Y, config['batch_size'])
            optimizer.zero_grad()
            y_hat = net(batch['x'])
            loss = loss_fn(y_hat, batch['y'])
            loss.backward()
            optimizer.step()

        timings.append(time.time() - start_time)
    
    return timings, loss.item()

# 4. Iterate over all configurations and record timings
timings = []
losses = []

for idx, config in enumerate(configs, 1):
    print(f"Running configuration {idx}/{len(configs)}: {config}")

    # run time_config in a new python process
    import multiprocessing as mp

    def run_time_config(config):
        timings, loss = time_config(config)
        return timings, loss

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_time_config, [config])

    results = results[0]

    print(f"Time taken: {results[0][9]:.2f} seconds\n")
    print(f"Loss: {results[1]:.2f}\n")
    timings.append(results[0])
    losses.append(results[1])

# each element of timings is a list of timings for each epoch
# convert them to a data.frame where timings[0] is the first row
timings = pd.DataFrame(timings, columns=[f"epoch_{i}" for i in range(1, 11)])


for config, loss in zip(configs, losses):
    config['loss'] = loss

# 6. Convert to pandas DataFrame for easier handling
df = pd.DataFrame(configs)

df = pd.concat([df, timings], axis=1)

name = f"result{os.getenv('CUDA_VISIBLE_DEVICES')}.csv"

# 7. Save the results to a file
# use projroot library to set it to ./2024-10-07/python/name.csv
from pyprojroot.here import here

output_path = str(here('2024-10-07/python/' + name))

df.to_csv(output_path, index=False)

print(f"Saved timings to {output_path}")