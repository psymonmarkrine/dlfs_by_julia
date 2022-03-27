function identity_function(x)
    return x
end

function step_function(x)
    return typeof(x)(x > 0)
end

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sigmoid_grad(x::Array{T}) where T <:Real
    return (1 .- sigmoid.(x)) .* sigmoid.(x)
end    

function relu(x)
    return maximum(0, x)
end

function relu_grad(x::Array{T}) where T <: Real
    grad = zero(x)
    grad[x.>=0] = 1
    return grad
end

function softmax(a::Array{T,2}) where T <:Real
    a = exp.(a .- maximum(a, dims=2)) # オーバーフロー対策
    return a ./ sum(a, dims=2)
end

function softmax(a)
    a = exp.(a .- maximum(a)) # オーバーフロー対策
    return a ./ sum(a)
end

function sum_squared_error(y, t)
    return 0.5 * sum((y-t).^2)
end

function cross_entropy_error(y::Vector{T}, t::Vector{Y}) where {T <: Real, Y <:Real}
    delta = 1.e-7
    return -t' * log.(y .+ delta)
end

function cross_entropy_error(y::Matrix{T}, t::Matrix{Y}) where {T <: Real, Y <:Real}
    batch_size = size(y, 1)
    return sum(cross_entropy_error.([y[i,:] for i=1:batch_size], [t[i,:] for i=1:batch_size])) / batch_size
end

function cross_entropy_error(y::Vector{T}, t::Y) where {T <: Real, Y <: Integer}
    delta = 1.e-7
    return -log.(y[t] .+ delta)
end

function cross_entropy_error(y::Matrix{T}, t::Vector{Y}) where {T <: Real, Y <: Integer}
    batch_size = size(y, 1)
    return sum(cross_entropy_error.([y[i,:] for i=1:batch_size], [t[i] for i=1:batch_size])) / batch_size
end

function softmax_loss(X, t)
    y = softmax(X)
    return cross_entropy_error(y, t)
end
