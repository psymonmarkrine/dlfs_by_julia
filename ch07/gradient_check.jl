import Statistics: mean

include("simple_convnet.jl") # SimpleConvNet

network = SimpleConvNet((1,10, 10), 
                        Dict("filter_num"=>10, "filter_size"=>3, "pad"=>0, "stride"=>1),
                        10, 10, 0.01)

X = reshape(rand(100), (1, 1, 10, 10))
T = [1]

grad_num = numerical_gradient(network, X, T)
grad = gradient(network, X, T)

for (key, val) = grad_num
    println("$key : $(mean(abs.(grad_num[key] - grad[key])))")
end
