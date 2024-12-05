library(torch)
library(here)
library(purrr)
library(data.table)
library(ignite)

test = FALSE

config_tbl = if (test) {
  list(
    # data
    n          = 100,
    p          = c(10, 20),
    # training parameters
    epochs     = 2,
    batch_size = c(16, 32),
    device     = c("cuda", "cpu"),
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
    p          = 1000,
    optimizer = c("adamw"),
    # training parameters
    epochs     = 2,
    batch_size = c(32),
    #device     = c("cuda", "cpu"),
    device     = c("cuda"),
    type       = c("free"),
    # jit compilation
    # network
    latent     = c(5000),
    n_layers   = c(16),
    #n_layers   = c(2),
    init_max_memory = c(FALSE)
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
  library(torch)
  nn_sequential2 = nn_module("nn_sequential2",
    initialize = function(...) {
      modules <- rlang::list2(...)
      for (i in seq_along(modules)) {
        self$add_module(name = i - 1, module = modules[[i]])
      }
    },
    forward = function(input) {
      self$tensors = list()
      for (module in private$modules_) {
        input <- module(input)
        torch::cuda_synchronize()
        self$tensors = append(self$tensors, input)
      }

      input
    }
  )
  # set cuda seed
  make_network = function(p, latent, n_layers) {
    layers = list(nn_linear(p, latent), nn_relu())
    for (i in seq_len(n_layers - 1)) {
        layers = c(layers, list(nn_linear(latent, latent), nn_relu()))
    }
    layers = c(layers, list(nn_linear(latent, 1)))

    do.call(nn_sequential2, args = layers)
  }

  library(ignite)
  if (config$init_max_memory) {
    # 11019 MiB
    x = torch_ones(10000 * 1024^2 / 4, device = config$device)
    x[1]
    rm(x)
    gc()
  }
  torch_manual_seed(123)
  ## convert the three lines below into R code, i.e. with rnorm and matrix multiplication
  X = torch_randn(config$n, config$p, device = config$device)
  beta = torch_randn(config$p, 1, device = config$device)
  Y = X$matmul(beta) + torch_randn(config$n, 1, device = config$device) * 0.1
  net = make_network(config$p, config$latent, config$n_layers)
  steps = ceiling(config$n / config$batch_size)
  if (config$device == "cuda") {
    net$cuda()
  }

  opt = ignite::optim_ignite_adamw(net$parameters, lr = 0.001)

  loss_fn = nn_mse_loss()

  do_step = if (config$type == "ignite") {
    net = jit_trace(net, X[1:config$batch_size, , drop = FALSE])
    loss_fn = jit_trace(loss_fn, net$forward(X[1:config$batch_size, , drop = FALSE]), Y[1:config$batch_size, , drop = FALSE])
    igniter = Igniter$new(
      network = net,
      optimizer = opt,
      loss_fn = loss_fn
    )

    function(input, target) {
      igniter$opt_step(list(input), target)[[1L]]
    }

  } else if (config$type == "free") {
    function(input, target) {
      opt$zero_grad()
      loss = loss_fn(net(input), target)
      loss$backward()
      opt$step()
      for (tensor in net$tensors) {
        torch:::torch_tensor_free(tensor)
      }
      loss
    }

  } else if (config$type == "jit") {
    net = jit_trace(net, X[1:config$batch_size, , drop = FALSE])
    function(input, target) {
      opt$zero_grad()
      loss = loss_fn(net(input), target)
      loss$backward()
      opt$step()
      loss
    }

  } else {
    function(input, target) {
      opt$zero_grad()
      loss = loss_fn(net(input), target)
      loss$backward()
      opt$step()
      loss
    }
  }

  get_batch = function(step, X, Y, batch_size) {
    list(
      x = X[seq((step - 1) * batch_size + 1, min(step * batch_size, config$n)), , drop = FALSE],
      y = Y[seq((step - 1) * batch_size + 1, min(step * batch_size, config$n)), , drop = FALSE]
    )
  }

  #timings = list()
  for (epoch in seq(config$epochs)) {
    #t0 = Sys.time()
    for (step in seq_len(steps)) {
      torch::cuda_synchronize()
      batch = get_batch(step, X, Y, config$batch_size)
      torch::cuda_synchronize()
      loss = do_step(batch$x, batch$y)
      torch::cuda_synchronize()
     }

  #timings = append(timings, list(as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }

  #list(
  #  timings = timings,
  #  loss = loss$item()
  #)
  return(NULL)
}


p = profvis::profvis({
  time(configs[[1]])}
 ,simplify = FALSE)

htmlwidgets::saveWidget(p, here("2024-11-22", "profile.html"), selfcontained = TRUE)

