import torch
import itertools
import time
import os
import pickle
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd

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
        'p': [100, 250, 500],
        'epochs': [20],
        'batch_size': [16, 128, 256],
        'device': ['cpu', 'cuda'],
        'jit': [True, False],
        'latent': [100, 500, 1000],
        'n_layers': [1, 5, 10]
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
    optimizer = Adam(net.parameters())
    loss_fn = nn.MSELoss()
    
    # Create DataLoader for batching
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Record start time
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            y_hat = net(batch_x)
            loss = loss_fn(y_hat, batch_y)
            loss.backward()
            optimizer.step()
    
    # Record end time
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed = end_time - start_time
    return elapsed, loss.item()

# 4. Iterate over all configurations and record timings
timings = []
for idx, config in enumerate(configs, 1):
    print(f"Running configuration {idx}/{len(configs)}: {config}")
    elapsed_time, loss = time_config(config)
    print(f"Time taken: {elapsed_time:.2f} seconds\n")
    print(f"Loss: {loss:.2f}\n")
    timings.append(elapsed_time)

# 5. Add timings to the configurations
for config, timing in zip(configs, timings):
    config['time'] = timing

# 6. Convert to pandas DataFrame for easier handling
df = pd.DataFrame(configs)

# 7. Save the results to a file
output_dir = Path("2024-10-07")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "pytorch2.csv"

# write to csv instead of pickle 

df.to_csv(output_path, index=False)

print(f"Saved timings to {output_path}")