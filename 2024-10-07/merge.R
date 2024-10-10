library(data.table)
library(here)

# There are result1 result2, result3 and result0
# add 
r0 = read.csv(here("2024-10-07/R/result0.csv"))
r1 = read.csv(here("2024-10-07/R/result1.csv"))
r2 = read.csv(here("2024-10-07/R/result2.csv"))
r3 = read.csv(here("2024-10-07/R/result3.csv"))

r = r0

r0$time = rowMeans(r0[, paste0("V", 6:10)])
r1$time = rowMeans(r1[, paste0("V", 6:10)])
r2$time = rowMeans(r2[, paste0("V", 6:10)])
r3$time = rowMeans(r3[, paste0("V", 6:10)])

r$loss = (r0$loss + r1$loss + r2$loss + r3$loss) / 4
r$time = (r0$time + r1$time + r2$time + r3$time) / 4
r$time_se = sqrt(((r0$time - r$time)^2 + (r1$time - r$time)^2 + (r2$time - r$time)^2 + (r3$time - r$time)^2) / 3) / sqrt(4)

setDT(r)
r$X = NULL
r = r[init_max_memory == FALSE, ]
r$init_max_memory = NULL
r$backend = "R"
# calculate differences between epochs, epoch_2 - epoch_1, epoch_3 - epoch_2, etc.

py0 = read.csv(here("2024-10-07/python/result0.csv"))
py1 = read.csv(here("2024-10-07/python/result1.csv"))
py2 = read.csv(here("2024-10-07/python/result2.csv"))
py2 = py2[py2$device == "cuda", ]
py3 = read.csv(here("2024-10-07/python/result3.csv"))
py3 = py3[py3$device == "cuda", ]

py = py0

for (i in paste0("epoch_", 1:10)) {
  py[[i]] = (py0[[i]], py1[[i]] + py2[[i]] + py3[[i]]) / 4
}

py$loss = (py0$loss +py1$loss + py2$loss + py3$loss) / 4

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

dt$epochs = 5

dt$n_batches = ceiling(dt$n / dt$batch_size) * dt$epochs

write.csv(dt, here("2024-10-07/analysis7.csv"), row.names = FALSE)



