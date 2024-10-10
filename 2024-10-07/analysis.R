library(data.table)
library(here)

r = read.csv(here("2024-10-07/rtorch5.csv"))
r$X = NULL
setDT(r)
r$init_max_memory = NULL
r$backend = "R"

py = read.csv(here("2024-10-07/pytorch5.csv"))
py$jit = as.logical(py$jit)
setDT(py)
py$backend = "Python"

# order the columns the same way
setcolorder(py, colnames(r))

# combine the data.tables
dt = rbind(r, py)

# Reshape dt from long to wide. I want one time column for backend R and one for backend Python
dt = dcast(dt, ... ~ backend, value.var = "time")

dt$n_batches = ceiling(dt$n / dt$batch_size) * dt$epochs

write.csv(dt, here("2024-10-07/analysis5.csv"), row.names = FALSE)



