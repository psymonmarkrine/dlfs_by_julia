using Base: Real
import HDF5
include("../dataset/mnist.jl") # load_mnist
include("../common/functions.jl") # sigmoid, softmax


function get_data()
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=true, flatten=true, one_hot_label=false)
    return x_test, t_test
end

function init_network()
    network = Dict()
    for k in ("W1", "W2", "W3", "b1", "b2", "b3")
        network[k] = HDF5.h5read("sample_weight.h5", k)
    end
    return network
end

function predict(network, x::Vector{T}) where T <: Real
    predict(network, x')
end

function predict(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = x * W1 .+ b1
    z1 = sigmoid.(a1)
    a2 = z1 * W2 .+ b2
    z2 = sigmoid.(a2)
    a3 = z2 * W3 .+ b3
    y = softmax(a3)

    return y
end

x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in 1:size(x, 1)
    y = predict(network, x[i,:])
    p = argmax(y[:]) # 最も確率の高い要素のインデックスを取得
    if (p) == typeof(p)(t[i])
        accuracy_cnt += 1
    end
end

print("Accuracy:$(accuracy_cnt / size(x,1))")
