library(data.table)
library(ggplot2)
library(here)

dt = fread(here("2024-10-07/analysis5.csv"))

tmp = dcast(dt[, -c("n_batches")], ... ~ epochs ,value.var = c("R", "Python"))
setnames(tmp, c("20", "40"), c("time_20", "time_40"))
tmp$Python_ratio = tmp$Python_40 / tmp$Python_20
tmp$R_ratio = tmp$R_40 / tmp$R_20

# For python the ratio between 40 and 20 epochs is 2, for R it is < 2
# Especially jitting becomes better with 40 epochs
tmp[, list(py = mean(Python_ratio), R = mean(R_ratio)), by = "jit"]

# -> Use 40 epochs now

dt = dt[epochs == 40, ]

# Effect of jitting?
# In python: no, in R it makes it slower
dt[, list(py = mean(Python), r = mean(R)), by = "jit"]

dt$ratio = dt$Python / dt$R

ggplot(dt[device == "cuda" & p == 500 & latent == 1000, ], aes(x = as.factor(n_layers), y = ratio, color = optimizer)) +
  geom_point() + 
  facet_grid(vars(jit), vars(batch_size)) + 
  labs(
    x = "Number of Layers",
    color = "JIT",
    title = "X-facet is Batch Size, Y-facet is jit, device is 'cuda', latent dim is 1000",
    y = "Ratio Runtime Pytorch / Rtorch"
  ) 

# What about the time per batch

dtlong = melt(dt, measure.vars = c("Python", "R"), value.name = "time", variable.name = "language")

dtlong$time_per_batch = dtlong$time / dtlong$n_batches

ggplot(dtlong[p == 500 & latent == 1000 & !jit & device == "cuda", ], aes(x = as.factor(batch_size), y = time_per_batch, color = language)) + 
  facet_grid(vars(n_layers), vars(optimizer), scales = "free_y") +
  geom_boxplot()
