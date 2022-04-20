include("../common/commons.jl")

module DeepConvNet_ch08

export DeepConvNet, predict, gradient, loss, accuracy, save_params, load_params

import OrderedCollections: OrderedDict

import HDF5
using  ..Layers
import ..Util: weight_init_randn

mutable struct DeepConvNet
    params::Dict
    layers
    last_layer
end

"""認識率99%以上の高精度なConvNet
ネットワーク構成は下記の通り
    conv - relu - conv- relu - pool -
    conv - relu - conv- relu - pool -
    conv - relu - conv- relu - pool -
    affine - relu - dropout - affine - dropout - softmax
"""
function DeepConvNet(input_dim=(1, 28, 28),
                conv_param_1 = Dict("filter_num"=>16, "filter_size"=>3, "pad"=>1, "stride"=>1),
                conv_param_2 = Dict("filter_num"=>16, "filter_size"=>3, "pad"=>1, "stride"=>1),
                conv_param_3 = Dict("filter_num"=>32, "filter_size"=>3, "pad"=>1, "stride"=>1),
                conv_param_4 = Dict("filter_num"=>32, "filter_size"=>3, "pad"=>2, "stride"=>1),
                conv_param_5 = Dict("filter_num"=>64, "filter_size"=>3, "pad"=>1, "stride"=>1),
                conv_param_6 = Dict("filter_num"=>64, "filter_size"=>3, "pad"=>1, "stride"=>1),
                hidden_size=50, output_size=10)
    # 重みの初期化===========
    # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
    # pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
    # weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値
    
    params = Dict()
    pre_channel_num = input_dim[1]
    for (idx, conv_param) = enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6])
        params["W$idx"] = weight_init_randn((conv_param["filter_num"], pre_channel_num, conv_param["filter_size"], conv_param["filter_size"]),"relu")
        params["b$idx"] = zeros(1, conv_param["filter_num"])
        pre_channel_num = conv_param["filter_num"]
    end
    params["W7"] = weight_init_randn((64*4*4, hidden_size), "relu")
    params["b7"] = zeros(1, hidden_size)
    params["W8"] = weight_init_randn((hidden_size, output_size), "relu")
    params["b8"] = zeros(1, output_size)

    # レイヤの生成===========
    layers = [Convolution(params["W1"], params["b1"], 
                          stride=conv_param_1["stride"], pad=conv_param_1["pad"]),
              Relu(),
              Convolution(params["W2"], params["b2"], 
                          stride=conv_param_2["stride"], pad=conv_param_2["pad"]),
              Relu(),
              Pooling(2, 2, stride=2),
              Convolution(params["W3"], params["b3"], 
                          stride=conv_param_3["stride"], pad=conv_param_3["pad"]),
              Relu(),
              Convolution(params["W4"], params["b4"],
                          stride=conv_param_4["stride"], pad=conv_param_4["pad"]),
              Relu(),
              Pooling(2, 2, stride=2),
              Convolution(params["W5"], params["b5"],
                          stride=conv_param_5["stride"], pad=conv_param_5["pad"]),
              Relu(),
              Convolution(params["W6"], params["b6"],
                          stride=conv_param_6["stride"], pad=conv_param_6["pad"]),
              Relu(),
              Pooling(2, 2, stride=2),
              Affine(params["W7"], params["b7"]),
              Relu(),
              Dropout(0.5),
              Affine(params["W8"], params["b8"]),
              Dropout(0.5)
             ]
    
    last_layer = SoftmaxWithLoss()

    return DeepConvNet(params, layers, last_layer)
end

function predict(self::DeepConvNet, x, train_flg=false)
    for layer = self.layers
        if typeof(layer) == Dropout
            x = forward(layer, x, train_flg)
        else
            x = forward(layer, x)
        end
    end
    return x
end

function loss(self::DeepConvNet, x, t)
    y = predict(self, x, true)
    return forward(self.last_layer, y, t)
end

function accuracy(self::DeepConvNet, x, t::Vector{T}, batch_size=100) where T <: Integer
    num_of_x = size(x, 1)
    index = collect(0:batch_size:(num_of_x + batch_size - 1))
    index[end] = num_of_x 
    l = length(index)
    return sum([accuracy(self, x[index[i-1]+1:index[i],:,:,:], t[index[i-1]+1:index[i]]) * (index[i]-index[i-1]) for i=2:l]) / num_of_x
end

function accuracy(self::DeepConvNet, x, t::Vector{T}) where T <: Integer
    y = predict(self, x)
    y = [i[2] for i=argmax(y, dims=2)]
    
    accuracy = sum(y .== eltype(y).(t)) / size(x, 1)
    return accuracy
end

function gradient(self::DeepConvNet, x, t)
    # forward
    loss(self, x, t)

    # backward
    dout = 1
    dout = backward(self.last_layer, dout)

    tmp_layers = reverse(self.layers)
    for layer = tmp_layers
        dout = backward(layer, dout)
    end

    # 設定
    grads = Dict()
    for (i, layer_idx) = enumerate((1, 3, 6, 8, 11, 13, 16, 19))
        grads["W$i"] = self.layers[layer_idx].dW
        grads["b$i"] = self.layers[layer_idx].db
    end

    return grads
end

function save_params(self::DeepConvNet, file_name="params.h5")
    for (key, val) = self.params
        HDF5.h5write(file_name, key, val)
    end
end

function load_params(self::DeepConvNet, file_name="params.h5")
    for (key, _) = self.params
        self.params[key] = HDF5.h5read(file_name, key)
    end

    for (i, key) = enumerate((1, 3, 6, 8, 11, 13, 16, 19))
        self.layers[key].W = self.params["W$(i)"]
        self.layers[key].b = self.params["b$(i)"]
    end
end

end # module DeepConvNet_ch08
