library(data.table)
library(here)

library(ggplot2)
library(data.table)

pytbl = fread("~/torch-profiling/2024-10-22/python/result1.csv")
pytbl$time = rowMeans(pytbl[, paste0("epoch_", 11:20)])
pytbl$se = apply(pytbl[, paste0("epoch_", 11:20)], 1, function(x) sd(x) / sqrt(length(x)))
pytbl$torchoptx = NA

pytbl[, paste0("epoch_", 1:20)] = NULL
pytbl$lang = "python"
tbl = fread("~/torch-profiling/2024-10-22/R/result1.csv")[, -1]
tbl$init_max_memory = NULL
tbl$lang = "R"
tbl$time = rowMeans(tbl[, paste0("V", 10:20)])
tbl$se = apply(tbl[, paste0("V", 10:20)], 1, function(x) sd(x) / sqrt(length(x)))
tbl[, paste0("V", 1:20)] = NULL
tbl = rbind(pytbl, tbl)
tbl

write.csv(tbl, "~/torch-profiling/2024-10-22/result.csv", row.names = FALSE)
