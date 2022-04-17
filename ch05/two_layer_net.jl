include("../common/commons.jl")

module TwoLayerNet_ch05

import  OrderedCollections: OrderedDict

import  ..Gradient: numerical_gradient
using   ..Layers

export  TwoLayerNet, predict, loss, accuracy, numerical_gradient, gradient

struct TwoLayerNet
    params::Dict
    layers::OrderedDict
    lastLayer
end

function TwoLayerNet(input_size::Integer, hidden_size::Integer, output_size::Integer, weight_init_std = 0.01)
    # 重みの初期化
    params = Dict(
        "W1" => weight_init_std * randn(input_size, hidden_size),
        "b1" => zeros(1, hidden_size),
        "W2" => weight_init_std * randn(hidden_size, output_size),
        "b2" => zeros(1, output_size)
    )
   
    # レイヤの生成
    layers = OrderedDict(
        "Affine1" => Affine(params["W1"], params["b1"]),
        "Relu1" => Relu(),
        "Affine2" => Affine(params["W2"], params["b2"])
    )

    lastLayer = SoftmaxWithLoss()

    return TwoLayerNet(params, layers, lastLayer)
end
        
function predict(self::TwoLayerNet, x)
    for (_, layer)=self.layers
        x = forward(layer, x)
    end
    return x
end

# x:入力データ, t:教師データ
function loss(self::TwoLayerNet, x, t)
    y = predict(self, x)
    return forward(self.lastLayer, y, t)
end

function accuracy(self::TwoLayerNet, x, t::Vector{T}) where T <: Integer
    y = predict(self, x)
    y = [i[2] for i=argmax(y, dims=2)]
    
    accuracy = sum(y .== eltype(y).(t)) / size(x, 1)
    return accuracy
end

function accuracy(self::TwoLayerNet, x, t)
    y = predict(self, x)
    y = argmax(y, dims=2)
    t = argmax(t, dims=2)
    
    accuracy = sum(y .== t) / size(x, 1)
    return accuracy
end

# x:入力データ, t:教師データ
function numerical_gradient(self::TwoLayerNet, x, t)
    loss_W = (W)->loss(self, x, t)
    
    grads = Dict(
        "W1" => numerical_gradient(loss_W, self.params["W1"]),
        "b1" => numerical_gradient(loss_W, self.params["b1"]),
        "W2" => numerical_gradient(loss_W, self.params["W2"]),
        "b2" => numerical_gradient(loss_W, self.params["b2"])
    )
    return grads
end

function gradient(self::TwoLayerNet, x, t)
    # forward
    loss(self, x, t)

    # backward
    dout = 1
    dout = backward(self.lastLayer, dout)
    
    layers = reverse(collect(values(self.layers)))
    for layer in layers
        dout = backward(layer, dout)
    end
    # 設定
    grads = Dict(
        "W1" => self.layers["Affine1"].dW,
        "b1" => self.layers["Affine1"].db,
        "W2" => self.layers["Affine2"].dW,
        "b2" => self.layers["Affine2"].db
    )
    return grads
end

end # module TwoLayerNet_ch05