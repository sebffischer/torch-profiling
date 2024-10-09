library(torch)
library(here)
library(purrr)

test = FALSE

config_tbl = if (test) {
  list(
    # data
    n          = 100,
    p          = c(10, 20),
    # training parameters
    epochs     = 2,
    batch_size = c(16, 32),
    device     = c("cpu", "cuda"),
    # jit compilation
    jit        = c(TRUE, FALSE),
    # network
    latent     = c(10, 50, 100),
    n_layers   = c(1, 5)
  ) |> expand.grid()
} else {
  list(
    # data
    n          = 2000,
    p          = c(100, 250, 500),
    # training parameters
    epochs     = 20,
    batch_size = c(16, 128, 256),
    device     = c("cpu", "cuda"),
    # jit compilation
    jit        = c(TRUE, FALSE),
    # network
    latent     = c(100, 500, 1000),
    n_layers   = c(1, 5, 10),
    init_max_memory = c(TRUE)
  ) |> expand.grid()
}


config_tbl$device = as.character(config_tbl$device)

# convert each row to a list but keep the names
configs = mlr3misc::map(1:nrow(config_tbl), function(i) {
  x = as.list(config_tbl[i, ])
  attr(x, "out.attrs") = NULL
  x
})


# I think it is important that we run this in a new R session each time because otherwise
# The cuda memory will depend on other experiments 
time = function(config) {
  make_network = function(p, latent, n_layers) {
    layers = list(nn_linear(p, latent), nn_relu())
    for (i in seq_len(n_layers - 1)) {
        layers = c(layers, list(nn_linear(latent, latent), nn_relu()))
    }
    layers = c(layers, list(nn_linear(latent, 1)))

    do.call(nn_sequential, args = layers)
  }

  library(torch)
  if (config$init_max_memory) {
    # 11019 MiB
    x = torch_ones(10000 * 1024^2 / 4, device = config$device)
    x[1]
    rm(x)
    gc()
  }
  ## convert the three lines below into R code, i.e. with rnorm and matrix multiplication
  X = torch_randn(config$n, config$p, device = config$device)
  beta = torch_randn(config$p, 1, device = config$device)
  Y = X$matmul(beta) + torch_randn(config$n, 1, device = config$device) * 0.1 
  net = make_network(config$p, config$latent, config$n_layers)
  steps = ceiling(config$n / config$batch_size)
  if (config$device == "cuda") {
    net$cuda()
  }

  if (config$jit) {
    net = jit_trace_module(net, forward = list(X[1:config$batch_size, , drop = FALSE]))
  }

  opt = optim_adam(net$parameters)


  get_batch = function(step, X, Y, batch_size) {
    list(
      x = X[seq((step - 1) * batch_size + 1, min(step * batch_size, config$n)), , drop = FALSE],
      y = Y[seq((step - 1) * batch_size + 1, min(step * batch_size, config$n)), , drop = FALSE]
    )
  }


  t0 = Sys.time()
  for (epoch in seq(1, config$epochs)) {
    for (step in seq_len(steps)) {
      opt$zero_grad()
      batch = get_batch(step, X, Y, config$batch_size)
      y_hat = net(batch$x)
      loss = nnf_mse_loss(y_hat, batch$y)
      loss$backward()
      opt$step()
    }
  }
  t1 = Sys.time()

  list(
    time = difftime(t1, t0, units = "secs"),
    loss = loss$item()
  )
}

timings = map_dbl(configs, function(config) {
  x = try(callr::r(time, args = list(config = config)))
  if (inherits(x, "try-error")) {
    print("error")
    return(NA)
  }
  print(x$loss)
  x$time
}, .progress = TRUE)

config_tbl$time = timings

write.csv(config_tbl, here("2024-10-07", "rtorch3.csv"))