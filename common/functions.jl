using Base: Real
function identity_function(x)
    return x
end

function step_function(x)
    return typeof(x)(x > 0)
end

function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function sigmoid_grad(x::T) where T <: Array{Y} where Y <:Real
    return (1 - sigmoid.(x)) * sigmoid.(x)
end    

function relu(x)
    return maximum(0, x)
end

function relu_grad(x::T) where T <: Array{Y} where Y <:Real
    grad = zero(x)
    grad[x>=0] = 1
    return grad
end

function softmax(a)
    a = exp.(a .- maximum(a, dims=2)) # オーバーフロー対策
    return a ./ sum(a, dims=2)
end

function sum_squared_error(y, t)
    return 0.5 * sum((y-t).^2)
end

function cross_entropy_error(y, t)
    if ndims(y) == 1
        t = reshape(t, 1, :)
        y = reshape(y, 1, :)
    end
    batch_size = size(y, 1)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if size(t) == size(y)
        t = argmax(t, dims=2)
        return -sum(log.(y[t] + 1e-7)) / batch_size
    end
    return -sum(log.([y[i, j] for (i,j) = enumerate(t)] + 1e-7)) / batch_size
end

function softmax_loss(X, t)
    y = softmax(X)
    return cross_entropy_error(y, t)
end