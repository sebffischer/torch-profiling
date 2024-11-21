options(torch.cuda_allocator_reserved_rate = 1.0)
options(torch.cuda_allocator_allocated_rate = 1.0)
options(torch.cuda_allocator_allocated_reserved_rate = 1.0)

library(torch)


#x = torch_randn(10000^2 * 15, device = "cuda")
#gc()

f = function() {
    e = new.env()
    reg.finalizer(e, function(e) {
        print("finalizing")
    })
    NULL
}


for (i in 1:200) {
    f()
}

allocs / (1024^3)

32 * 10000^2 / 1024^2
32 * 1000^2 / 1024^2
32 * 100^2 / 1024^2

# single alloc

# 100^2: 659MiB
# 1000^2: 677MiB
# 10000^2: 1039MiB

# 10 allocs: 

# 100^2: 661MiB
# 1000^2: 1057MiB
# 10000^2: 8289MiB 
