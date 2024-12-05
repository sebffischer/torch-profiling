library(batchtools)
library(mlr3misc)

reg = loadRegistry("~/prof1", writeable = TRUE)

results = map(findDone()[[1]], loadResult)
