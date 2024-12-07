time_rtorch = function(epochs, batch_size, n_layers, latent, n, p, optimizer, device, num_threads, num_interop_threads, type, seed) {
  library(torch)
  library(ignite)
  library(mlr3torch)
  torch_manual_seed(seed)
  do_jit = type %in% c("torchoptx_jit", "ignite", "mlr3torch_ignite")
  ignite_opt = type %in% c("torchoptx", "torchoptx_jit", "ignite", "mlr3torch_ignite")

  make_network = function(p, latent, n_layers) {
    layers = list(nn_linear(p, latent), nn_relu())
    for (i in seq_len(n_layers - 1)) {
        layers = c(layers, list(nn_linear(latent, latent), nn_relu()))
    }
    layers = c(layers, list(nn_linear(latent, 1)))

    net = do.call(nn_sequential, args = layers)
    if (do_jit) {
      net = jit_trace(net, torch_randn(1, p))
    }
    net
  }

  X = torch_randn(n, p, device = device)
  beta = torch_randn(p, 1, device = device)
  Y = X$matmul(beta) + torch_randn(n, 1, device = device) * 0.1

  net = make_network(p, latent, n_layers)
  net$to(device = device)

  opt_class = if (ignite_opt) {
    switch(optimizer,
      adamw = ignite::optim_ignite_adamw,
      sgd = ignite::optim_ignite_sgd
    )
  } else {
    switch(optimizer,
      adamw = torch::optim_adamw,
      sgd = torch::optim_sgd
    )
  }

  loss_fn = if (type %in% c("ignite", "torchoptx_jit", "mlr3torch_ignite")) {
    jit_trace(nn_mse_loss(), with_no_grad(net$forward(X[1:batch_size, , drop = FALSE])), Y[1:batch_size, , drop = FALSE])
  } else {
    nn_mse_loss()
  }

  opt = opt_class(net$parameters, lr = 0.001)
  steps = ceiling(n / batch_size)

  # this function should train the network for the given number of epochs and return the final training loss
  train_run = if (type != "mlr3torch") {
    do_step = if(type %in% c("base", "torchoptx", "torchoptx_jit")) {
      # the only difference is how optimizer was defined and whether the network was trace-jitted
      function(input, target) {
        opt$zero_grad()
        loss = loss_fn(net(input), target)
        loss$backward()
        opt$step()
        loss
      }
    } else if (type == "ignite") {
      igniter = Igniter$new(
        network = net,
        optimizer = opt,
        loss_fn = loss_fn
      )
      function(input, target) {
        igniter$opt_step(list(input), target)[[1L]]
      }
    } else {
      stop("Unknown type")
    }

    get_batch = function(step, X, Y, batch_size) {
      list(
        x = X[seq((step - 1) * batch_size + 1, min(step * batch_size, n)), , drop = FALSE],
        y = Y[seq((step - 1) * batch_size + 1, min(step * batch_size, n)), , drop = FALSE]
      )
    }

    function() {
      for (epoch in seq(epochs)) {
        for (step in seq_len(steps)) {
          batch = get_batch(step, X, Y, batch_size)
          loss = do_step(batch$x, batch$y)
        }
      }
      return(loss$item())
    }

  } else if (type == "mlr3torch") {
    extract_loss = torch_callback("extract_loss",
      state_dict = function() {
        self$ctx$last_loss
      },
      load_state_dict = function(state_dict) {
        self$ctx$last_loss = state_dict
      }
    )

    # TODO: Do we need `tensor_dataset = TRUE` here for a fair comparison?
    learner = LearnerTorchModel$new(
      network = net,
      task_type = "regr",
      optimizer = as_torch_optimizer(opt_class),
      ingress_tokens = list(
        x = TorchIngressToken(
          features = "x",
          batchgetter = mlr3torch:::batchgetter_lazy_tensor,
          shape = c(NA, p)
        )
      ),
      callbacks = extract_loss,
      loss = t_loss("mse")
    )
    learner$param_set$set_values(
      opt.lr = 0.001,
      device = device,
      jit_trace = TRUE,
      epochs = epochs,
      drop_last = FALSE,
      batch_size = batch_size
    )

    task = as_task_regr(data.table(
      x = as_lazy_tensor(X),
      y = as.numeric(Y)
    ), target = "y")

    function() {
      learner$train(task)$model$callbacks$extract_loss
    }
  }

  t0 = Sys.time()
  loss = train_run()
  t = difftime(Sys.time(), t0, units = "secs")

  list(time = t, last_loss = loss)
}

if (FALSE) {
    library(torch)
    library(ignite)
    library(mlr3torch)
    args = list(
        epochs = 1,
        batch_size = 1,
        n_layers = 1,
        latent = 1,
        n = 1,
        p = 1,
        optimizer = "adamw",
        device = "cpu",
        num_threads = 1,
        num_interop_threads = 1,
        seed = 1
    )
    do.call(time_rtorch, c(args, list(type = "base")))
    do.call(time_rtorch, c(args, list(type = "ignite")))
    do.call(time_rtorch, c(args, list(type = "torchoptx")))
    do.call(time_rtorch, c(args, list(type = "torchoptx_jit")))
    do.call(time_rtorch, c(args, list(type = "mlr3torch")))
}
