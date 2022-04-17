include("../common/functions.jl")
include("../common/gradient.jl")

import  .Functions: softmax, cross_entropy_error
import  .Gradient: numerical_gradient


struct SimpleNet
    W::Matrix{T} where T <: AbstractFloat
end

SimpleNet() = SimpleNet(randn(2,3))

function predict(self::SimpleNet, x)
    return x*self.W
end

function loss(self::SimpleNet, x, t)
    z = predict(self, x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)
    
    return loss
end

x = [0.6 0.9]
t = [0.0 0.0 1.0]

net = SimpleNet([0.47355232 0.9977393 0.84668094; 0.85557411 0.03563661 0.69422093])

f = (w)->loss(net, x, t)
dW = numerical_gradient(f, net.W)

display(dW)
