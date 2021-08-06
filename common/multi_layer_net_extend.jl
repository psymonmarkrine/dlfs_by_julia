import OrderedCollections: OrderedDict

include("layers.jl")
include("gradient.jl") # numerical_gradient


mutable struct MultiLayerNetExtend
    input_size::Integer
    output_size::Integer
    hidden_size_list::Vector{T} where T <: Integer
    hidden_layer_num::Integer
    weight_decay_lambda::AbstractFloat
    params::IdDict
    layers::OrderedDict
    last_layer
    use_dropout::Bool
    use_batchnorm::Bool
end

function MultiLayerNetExtend(input_size::Integer, hidden_size_list::Vector{T}, output_size::Integer;
    activation="relu", weight_init_std="relu", weight_decay_lambda::AbstractFloat=0.0, 
    use_dropout::Bool = false, dropout_ration::AbstractFloat = 0.5, use_batchnorm::Bool=false) where T <: Integer
    """拡張版の全結合による多層ニューラルネットワーク
    
    Weiht Decay、Dropout、Batch Normalizationの機能を持つ
    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : "relu" or "sigmoid"
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        "relu"または"he"を指定した場合は「Heの初期値」を設定
        "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration : Dropoutの割り合い
    use_batchNorm: Batch Normalizationを使用するかどうか
    """
    hidden_layer_num = length(hidden_size_list)
    last_layer = SoftmaxWithLoss()
    self = MultiLayerNetExtend(input_size, output_size, hidden_size_list, hidden_layer_num, weight_decay_lambda,
                               IdDict(), OrderedDict(), last_layer, use_dropout, use_batchnorm)

    # 重みの初期化
    __init_weight(self, weight_init_std)

    # レイヤの生成
    activation_layer = IdDict("sigmoid"=>Sigmoid, "relu"=>Relu)
    for idx = 1:self.hidden_layer_num
        self.layers["Affine$idx"] = Affine(self.params["W$idx"],
                                           self.params["b$idx"])
        if self.use_batchnorm
            self.params["beta$idx"] = zeros(1, hidden_size_list[idx])
            self.params["gamma$idx"] = ones(1, hidden_size_list[idx])
            self.layers["BatchNorm$idx"] = BatchNormalization(self.params["gamma$idx"], self.params["beta$idx"])
        end
        self.layers["Activation_function$idx"] = activation_layer[activation]()
        
        if self.use_dropout
            self.layers["Dropout$idx"] = Dropout(dropout_ration)
        end
    end
    idx = self.hidden_layer_num + 1
    self.layers["Affine$idx"] = Affine(self.params["W$idx"], self.params["b$idx"])
    return self
end

function __init_weight(self::MultiLayerNetExtend, weight_init_std)
    """重みの初期値設定
    Parameters
    ----------
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        "relu"または"he"を指定した場合は「Heの初期値」を設定
        "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定
    """
    all_size_list = vcat(self.input_size, self.hidden_size_list, self.output_size)
    for idx = 1:length(all_size_list)-1
        scale = weight_init_std
        if lowercase("$weight_init_std") in ("relu", "he")
            scale = sqrt(2.0 / all_size_list[idx])  # ReLUを使う場合に推奨される初期値
        elseif lowercase("$weight_init_std") in ("sigmoid", "xavier")
            scale = sqrt(1.0 / all_size_list[idx])  # sigmoidを使う場合に推奨される初期値
        end
        self.params["W$idx"] = scale * randn(all_size_list[idx], all_size_list[idx+1])
        self.params["b$idx"] = zeros(1, all_size_list[idx+1])
    end
end

function predict(self::MultiLayerNetExtend, x, train_flg=false)
    for (key,layer) = self.layers
        if "Dropout" == key || "BatchNorm" == key
            x = forward(layer, x, train_flg)
        else
            x = forward(layer, x)
        end
    end
    return x
end

function loss(self::MultiLayerNetExtend, x, t, train_flg=false)
    """損失関数を求める
    引数のxは入力データ、tは教師ラベル
    """
    y = predict(self, x, train_flg)

    weight_decay = 0.0
    for idx = 1:self.hidden_layer_num+1
        W = self.params["W$idx"]
        weight_decay += 0.5 * self.weight_decay_lambda * sum(W.^2)
    end
    return forward(self.last_layer, y, t) + weight_decay
end

function accuracy(self::MultiLayerNetExtend, x, t::Vector{T}) where T <: Integer
    y = predict(self, x, false)
    y = [i[2] for i=argmax(y, dims=2)]
    
    accuracy = sum(y .== eltype(y).(t)) / size(x, 1)
    return accuracy
end

function accuracy(self::MultiLayerNetExtend, x, t)
    y = predict(self, x, false)
    y = argmax(y, dims=2)
    t = argmax(t, dims=2)
    
    accuracy = sum(y .== t) / size(x, 1)
    return accuracy
end

function numerical_gradient(self::MultiLayerNetExtend, x, t)
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
    loss_W = (W)->loss(self, x, t, false)

    grads = Dict()
    for idx in 1:self.hidden_layer_num+1
        grads["W$idx"] = numerical_gradient(loss_W, self.params["W$idx"])
        grads["b$idx"] = numerical_gradient(loss_W, self.params["b$idx"])

        if self.use_batchnorm && idx != self.hidden_layer_num+1
            grads["gamma$idx"] = numerical_gradient(loss_W, self.params["gamma$idx"])
            grads["beta$idx"] = numerical_gradient(loss_W, self.params["beta$idx"])
        end
    end
    return grads
end

function gradient(self::MultiLayerNetExtend, x, t)
    # forward
    loss(self, x, t, true)

    # backward
    dout = 1
    dout = backward(self.last_layer, dout)

    layers = reverse(collect(values(self.layers)))
    for layer in layers
        dout = backward(layer, dout)
    end

    # 設定
    grads = Dict()
    for idx =1:self.hidden_layer_num+1
        grads["W$idx"] = self.layers["Affine$idx"].dW + self.weight_decay_lambda * self.layers["Affine$idx"].W
        grads["b$idx"] = self.layers["Affine$idx"].db

        if self.use_batchnorm && idx != self.hidden_layer_num+1
            grads["gamma$idx"] = self.layers["BatchNorm$idx"].dgamma
            grads["beta$idx"] = self.layers["BatchNorm$idx"].dbeta
        end
    end
    return grads
end
