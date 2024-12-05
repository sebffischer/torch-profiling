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

py0$time = rowMeans(py0[, paste0("epoch_", 6:10)])
py1$time = rowMeans(py1[, paste0("epoch_", 6:10)])
py2$time = rowMeans(py2[, paste0("epoch_", 6:10)])
py3$time = rowMeans(py3[, paste0("epoch_", 6:10)])

py = py0

py$loss = (py0$loss + py1$loss + py2$loss + py3$loss) / 4
py$time = (py0$time + py1$time + py2$time + py3$time) / 4
py$time_se = sqrt(((py0$time - py$time)^2 + (py1$time - py$time)^2 + (py2$time - py$time)^2 + (py3$time - py$time)^2) / 3) / sqrt(4)


py$jit = as.logical(py$jit)
py$backend = "Python"

#remove epoch_1, ... epoch_10
for (i in 1:10) {
  py[[paste0("epoch_", i)]] = NULL
}
for (i in 1:10) {
  r[[paste0("V", i)]] = NULL
}


setcolorder(py, colnames(r))
# order the columns the same way


# combine the data.tables
dt = rbind(r, py)

# Reshape dt from long to wide. I want one time column for backend R and one for backend Python
dt = dcast(dt, ... ~ backend, value.var = c("time", "loss", "time_se"))

dt$epochs = 5

dt$n_batches = ceiling(dt$n / dt$batch_size) * dt$epochs


# i want Pytorch_time, R_time, Pytorch_loss, R_loss in long format
# but time should be one column and loss should be one column

write.csv(dt, here("2024-10-07/analysis7.csv"), row.names = FALSE)



