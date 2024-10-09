library(data.table)
library(here)

r = read.csv(here("2024-10-07/rtorch2.csv"))
r$X = NULL
setDT(r)
r$backend = "R"
r$init_max_memory = NULL

py = read.csv(here("2024-10-07/pytorch2.csv"))
py$jit = as.logical(py$jit)
setDT(py)
py$backend = "Python"

# order the columns the same way
setcolorder(py, colnames(r))

py = py[order(list(n, p, epochs, batch_size, device, jit, latent, n_layers))]
r = r[order(list(n, p, epochs, batch_size, device, jit, latent, n_layers))]

# combine the data.tables
dt = rbind(r, py)

# Reshape dt from long to wide. I want one time column for backend R and one for backend Python
dt = dcast(dt, ... ~ backend, value.var = "time")

dt$ratio = dt$R / dt$Python

write.csv(dt, here("2024-10-07/analysis.csv"), row.names = FALSE)


