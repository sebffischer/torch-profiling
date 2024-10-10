library(data.table)
library(here)

r = read.csv(here("2024-10-07/R/result1.csv"))
setDT(r)
r$X = NULL
setnames(r, paste0("V", 1:10), paste0("epoch_", 1:10))
r = r[init_max_memory == FALSE, ]
r$init_max_memory = NULL
r$backend = "R"
# calculate differences between epochs, epoch_2 - epoch_1, epoch_3 - epoch_2, etc.
r$epoch_diff_1 = r$epoch_1
for (i in 2:10) {
  r[[paste0("epoch_diff_", i)]] = r[[paste0("epoch_", i)]] - r[[paste0("epoch_", i - 1)]]
}
# remove epoch_1 to epoch_10
r[, paste0("epoch_", 1:10) := NULL]
setnames(r, paste0("epoch_diff_", 1:10), paste0("epoch_", 1:10))
r$time = rowMeans(r[, paste0("epoch_", 6:10)])
r[, paste0("epoch_", 1:10) := NULL]
# average epochs 6 to 10

py = read.csv(here("2024-10-07/python/result2.csv"))
py$jit = as.logical(py$jit)
setDT(py)
py = py[py$device == "cuda", ]
py$backend = "Python"

# do the same for pytorch
py$time = rowMeans(py[, paste0("epoch_", 6:10)])
# calculate differences
py$epoch_diff_1 = py$epoch_1
for (i in 2:10) {
  py[[paste0("epoch_diff_", i)]] = py[[paste0("epoch_", i)]] - py[[paste0("epoch_", i - 1)]]
}
# set epochs to null
py[, paste0("epoch_", 1:10) := NULL]
# change names
setnames(py, paste0("epoch_diff_", 1:10), paste0("epoch_", 1:10))
py$time = rowMeans(py[, paste0("epoch_", 6:10)])
py[, paste0("epoch_", 1:10) := NULL]





setcolorder(py, colnames(r))
# order the columns the same way


# combine the data.tables
dt = rbind(r, py)

head

# Reshape dt from long to wide. I want one time column for backend R and one for backend Python
dt = dcast(dt, ... ~ backend, value.var = c("time", "loss"))

dt$n_batches = ceiling(dt$n / dt$batch_size) * dt$n

write.csv(dt, here("2024-10-07/analysis7.csv"), row.names = FALSE)



