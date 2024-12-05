library(batchtools)

reg = loadRegistry("~/prof1", writeable = TRUE)

jt = unwrap(getJobTable())
submitJobs(11)
