include("../common/commons.jl")

module SimpleConvNet_ch07

export SimpleConvNet, predict, loss, accuracy, numerical_gradient, gradient,save_params,load_params

import OrderedCollections: OrderedDict

import HDF5

import ..Gradient: numerical_gradient
using  ..Layers
import ..Util: weight_init_randn

mutable struct SimpleConvNet
    params::Dict
    layers::OrderedDict
    last_layer 
end
"""単純なConvNet
conv - relu - pool - affine - relu - affine - softmax

Parameters
----------
input_size : 入力サイズ（MNISTの場合は784）
hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
output_size : 出力サイズ（MNISTの場合は10）
activation : "relu" or "sigmoid"
weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
    "relu"または"he"を指定した場合は「Heの初期値」を設定
    "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定
"""
function SimpleConvNet(input_dim=(1, 28, 28), 
                       conv_param=Dict("filter_num"=>30, "filter_size"=>5, "pad"=>0, "stride"=>1),
                       hidden_size=100, output_size=10, weight_init_std=0.01)
    filter_num = conv_param["filter_num"]
    filter_size = conv_param["filter_size"]
    filter_pad = conv_param["pad"]
    filter_stride = conv_param["stride"]
    input_size = input_dim[2]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = Integer(round(filter_num * (conv_output_size/2) * (conv_output_size/2)))
    
    # 重みの初期化
    params = Dict(
        "W1" => weight_init_randn((filter_num, input_dim[1], filter_size, filter_size), weight_init_std),
        "b1" => zeros(1, filter_num),
        "W2" => weight_init_randn((pool_output_size, hidden_size), weight_init_std),
        "b2" => zeros(1, hidden_size),
        "W3" => weight_init_randn((hidden_size, output_size), weight_init_std),
        "b3" => zeros(1, output_size)
    )

    # レイヤの生成
    layers = OrderedDict(
        "Conv1" => Convolution(params["W1"], params["b1"], stride=conv_param["stride"], pad=conv_param["pad"]),
        "Relu1" => Relu(),
        "Pool1" => Pooling(2, 2, stride=2),
        "Affine1" => Affine(params["W2"], params["b2"]),
        "Relu2" => Relu(),
        "Affine2" => Affine(params["W3"], params["b3"]),
    )
    last_layer = SoftmaxWithLoss()

    return SimpleConvNet(params, layers, last_layer)
end

function predict(self::SimpleConvNet, x)
    for (_,layer) = self.layers
        x = forward(layer, x)
    end
    return x
end

function loss(self::SimpleConvNet, x, t)
    """損失関数を求める
    引数のxは入力データ、tは教師ラベル
    """
    y = predict(self, x)
    return forward(self.last_layer, y, t)
end

function accuracy(self::SimpleConvNet, x, t::Vector{T}, batch_size=100) where T <: Integer
    index = 1:batch_size:size(x, 1)
    index[end] = size(x, 1)
    l = length(index)
    return sum([accuracy(self, x[index[i-1]:index[i],:], t[index[i-1]:index[i]]) for i=2:l]) / l
end

function accuracy(self::SimpleConvNet, x, t::Vector{T}) where T <: Integer
    y = predict(self, x)
    y = [i[2] for i=argmax(y, dims=2)]
    
    accuracy = sum(y .== eltype(y).(t)) / size(x, 1)
    return accuracy
end

function accuracy(self::SimpleConvNet, x, t, batch_size=100) where T <: Integer
    index = 1:batch_size:size(x, 1)
    index[end] = size(x, 1)
    l = length(index)
    return sum([accuracy(self, x[index[i-1]:index[i],:], t[index[i-1]:index[i],:]) for i=2:l]) / l
end

function accuracy(self::SimpleConvNet, x, t)
    y = predict(self, x)
    y = argmax(y, dims=2)
    t = argmax(t, dims=2)
    
    accuracy = sum(y .== t) / size(x, 1)
    return accuracy
end

function numerical_gradient(self::SimpleConvNet, x, t)
    """勾配を求める（数値微分）
    Parameters
    ----------
    x : 入力データ
    t : 教師ラベル
    Returns
    -------
    各層の勾配を持ったディクショナリ変数
        grads["W1"]、grads["W2"]、...は各層の重み
        grads["b1"]、grads["b2"]、...は各層のバイアス
    """
    loss_w = (w)->loss(self, x, t)

    grads = Dict()
    for idx = (1, 2, 3)
        grads["W$idx"] = numerical_gradient(loss_w, self.params["W$idx"])
        grads["b$idx"] = numerical_gradient(loss_w, self.params["b$idx"])
    end
    return grads
end

function gradient(self::SimpleConvNet, x, t)
    """勾配を求める（誤差逆伝搬法）
    Parameters
    ----------
    x : 入力データ
    t : 教師ラベル
    Returns
    -------
    各層の勾配を持ったディクショナリ変数
        grads["W1"]、grads["W2"]、...は各層の重み
        grads["b1"]、grads["b2"]、...は各層のバイアス
    """
    # forward
    loss(self, x, t)

    # backward
    dout = 1
    dout = backward(self.last_layer, dout)

    layers = reverse(collect(values(self.layers)))
    for layer = layers
        dout = backward(layer, dout)
    end

    # 設定
    grads = Dict(
        "W1" => self.layers["Conv1"].dW,
        "b1" => self.layers["Conv1"].db,
        "W2" => self.layers["Affine1"].dW,
        "b2" => self.layers["Affine1"].db,
        "W3" => self.layers["Affine2"].dW,
        "b3" => self.layers["Affine2"].db
    )
    return grads
end

function save_params(self::SimpleConvNet, file_name="params.h5")
    params = Dict()
    for (key, val) = self.params
        HDF5.h5write(file_name, key, val)
    end
end

function load_params(self, file_name="params.h5")
    for (key, _) = self.params
        self.params[key] = HDF5.h5read(file_name, key)
    end
    for (i, key) = enumerate(["Conv1", "Affine1", "Affine2"])
        self.layers[key].W = self.params["W$(i)"]
        self.layers[key].b = self.params["b$(i)"]
    end
end

end # module SimpleConvNet_07
