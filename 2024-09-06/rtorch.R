library(torch)
library(here)
#library(torchoptx)

p = 100
steps = 1000

n = 1000

X = torch_randn(n, p, device = "cuda")
beta = torch_randn(p, 1, device = "cuda")
Y = X$matmul(beta) 

latent = 5000

net = nn_sequential(
  nn_linear(p, latent),
  nn_relu(),
  nn_linear(latent, 1)
)


net$cuda()
#for (par in net$parameters) par$requires_grad_(FALSE)
opt = optim_adam(net$parameters)
netf = jit_trace_module(net, forward = list(X))
#backward_jit = jit_trace(backward, loss)

t1 = Sys.time()

#p = profvis::profvis({
  for (i in 1:steps) {
    opt$zero_grad() 
    Y_hat = netf(X)
    loss = nnf_mse_loss(Y_hat, Y)
    loss$backward()
    opt$step()
    torch:::torch_free(Y_hat)
    torch:::torch_free(loss)
  } 
# ,simplify = FALSE)

t2 = Sys.time()

#htmlwidgets::saveWidget(p, here("2024-09-06", "profile.html"), selfcontained = TRUE)

print(paste0("Total time: ", t2 - t1))

