library(batchtools)
library(mlr3misc)

reg = makeExperimentRegistry(
  file.dir = "~/prof1",
  packages = c("checkmate", "reticulate"),
  work.dir = "~/torch-profiling/2024-11-22"
)


# this defines the time_pytorch function
source("~/torch-profiling/2024-11-22/time_rtorch.R")


if (FALSE) {
  args = list(
    epochs = 1L,
    batch_size = 1L,
    n_layers = 1L,
    n = 1L,
    p = 1L,
    latent = 1L,
    optimizer = "adamw",
    device = "cpu",
    num_threads = 1L,
    num_interop_threads = 1L,
    seed = 1L,
    type = "base"
  )
  do.call(time_rtorch, args)
}

batchExport(list(
  time_rtorch = time_rtorch
))


# The algorithm should return the total runtime needed for training, the SD, but also the performance of the training losses so we know it is all working
addProblem("runtime_train",
  data = NULL,
  fun = function(epochs, batch_size, n_layers, latent, n, p, optimizer, device, num_threads, num_interop_threads, ...) {
    problem = list(
      epochs = assert_int(epochs),
      batch_size = assert_int(batch_size),
      n_layers = assert_int(n_layers),
      latent = assert_int(latent),
      n = assert_int(n),
      p = assert_int(p),
      optimizer = assert_choice(optimizer, c("adamw", "sgd")),
      device = assert_choice(device, c("cpu", "cuda")),
      # cpu-specific parameters
      num_threads = assert_int(num_threads, null.ok = TRUE),
      num_interop_threads = assert_int(num_interop_threads, null.ok = TRUE)
    )
    if (device == "cuda") {
      assert_true(num_threads == 1)
      assert_true(num_interop_threads == 1)
    }
    problem
  }
)

# pytorch needs to be submitted with an active pytorch environment
# as I otherwise get the: OSError: libmkl_intel_lp64.so.2: cannot open shared object file: 
addAlgorithm("pytorch",
  fun = function(instance, job, ...) {
    f = function(...) {
      reticulate::use_condaenv("pytorch")
      reticulate::source_python("~/torch-profiling/2024-11-22/time_pytorch.py")
      time_pytorch(...)
    }
    callr::r(f, args = c(instance, list(seed = job$seed)))
  }
)

# type can be: "base", "torchoptx", "torchoptx_jit", "ignite", and "mlr3torch_ignite" and indicates how the model is fit.
addAlgorithm("rtorch",
  fun = function(instance, job, type, ...) {
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, type = type)))
  }
)


problem_design = list(
  n          = 2000,
  p          = 1000,
  optimizer = c("adamw"),
  # training parameters
  epochs     = 20,
  batch_size = c(32, 64, 128),
  device     = c("cuda", "cpu"),
  # jit compilation
  # network
  latent     = c(500, 1000, 2000),
  n_layers   = c(2, 4, 8, 16, 32),
  num_interop_threads = 1,
  num_threads = 1
) |> expand.grid(stringsAsFactors = FALSE)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(type = c("base", "torchoptx", "torchoptx_jit", "ignite", "mlr3torch")),
    pytorch = data.frame()
  )
)
