module Layers

export  forward, backward, 
        Relu, Sigmoid, SoftmaxWithLoss,
        Dropout, BatchNormalization,
        Affine, Convolution, Pooling

import  Statistics: mean

using   ..Functions
import  ..Util: im2col, col2im

mutable struct Relu
    mask
end

Relu() = Relu(nothing)

function forward(self::Relu, x)
    self.mask = (x .<= 0)
    out = x
    out[self.mask] .= 0

    return out
end

function backward(self::Relu, dout)
    dout[self.mask] .= 0
    dx = dout

    return dx
end


mutable struct Sigmoid
    out
end

Sigmoid() = Sigmoid(nothing)

function forward(self::Sigmoid, x)
    out = sigmoid.(x)
    self.out = out
    return out
end

function backward(self, dout)
    dx = dout .* (1.0 .- self.out) .* self.out

    return dx
end


mutable struct Affine
    W
    b
    x
    original_x_shape
    dW
    db
end

function Affine(W, b)
    if ndims(b)==1
        b = reshape(b, 1, :)
    end
    Affine(W, b, nothing, nothing, nothing, nothing)
end

function forward(self::Affine, x)
    # テンソル対応
    self.original_x_shape = size(x)
    self.x = reshape(x, self.original_x_shape[1], :)

    out = self.x * self.W .+ self.b

    return out
end

function backward(self::Affine, dout)
    dx = dout * self.W'
    self.dW = self.x' * dout
    self.db = sum(dout, dims=1)
    
    dx = reshape(dx, self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
    return dx
end


mutable struct SoftmaxWithLoss
    loss
    y # softmaxの出力
    t # 教師データ
end

SoftmaxWithLoss() = SoftmaxWithLoss(nothing, nothing, nothing)

function forward(self::SoftmaxWithLoss, x, t)
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    
    return self.loss
end

function backward(self::SoftmaxWithLoss, dout=1)
    batch_size = size(self.t, 1)
    if length(self.t) == length(self.y) # 教師データがone-hot-vectorの場合
        dx = (self.y - self.t) ./ batch_size
    else
        dx = copy(self.y)
        for (i,j) = enumerate(self.t)
            dx[i, j] -= 1
        end
        dx = dx ./ batch_size
    end
    return dx
end


mutable struct Dropout
    dropout_ratio::AbstractFloat
    mask
end

function Dropout(dropout_ratio=0.5)
    """
    http://arxiv.org/abs/1207.0580
    """
    return Dropout(dropout_ratio, nothing)
end

function forward(self::Dropout, x, train_flg=true)
    if train_flg
        self.mask = rand(eltype(x), size(x)) .> self.dropout_ratio
        return x .* self.mask
    else
        return x * (1.0 - self.dropout_ratio)
    end
end

function backward(self::Dropout, dout)
    return dout .* self.mask
end


mutable struct BatchNormalization
    gamma
    beta
    momentum
    input_shape  # Conv層の場合は4次元、全結合層の場合は2次元
    # テスト時に使用する平均と分散
    running_mean
    running_var
    # backward時に使用する中間データ
    batch_size
    xc
    xn
    std
    dgamma
    dbeta
end

function BatchNormalization(gamma, beta; momentum=0.9, running_mean=nothing, running_var=nothing)
    """
    http://arxiv.org/abs/1502.03167
    """
    return BatchNormalization(gamma, beta, momentum, nothing, running_mean, running_var, nothing, nothing, nothing, nothing, nothing, nothing)
end

function forward(self::BatchNormalization, x, train_flg=true)
    self.input_shape = size(x)
    if ndims(x) != 2
        N, C, H, W = size(x)
        x = reshape(x, N, :)
    end

    out = __forward(self, x, train_flg)
    
    return reshape(out, self.input_shape)
end
        
function __forward(self::BatchNormalization, x, train_flg)
    if isnothing(self.running_mean)
        N, D = size(x)
        self.running_mean = zeros(1, D)
        self.running_var = zeros(1, D)
    end
    if train_flg
        mu = mean(x, dims=1)
        xc = x .- mu
        var = mean(xc.^2, dims=1)
        std = sqrt.(var .+ 10e-7)
        xn = xc ./ std
        
        self.batch_size = size(x, 1)
        self.xc = xc
        self.xn = xn
        self.std = std
        self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
    else
        xc = x .- self.running_mean
        xn = xc ./ sqrt(self.running_var .+ 10e-7)
    end
    out = self.gamma .* xn .+ self.beta 
    return out
end

function backward(self::BatchNormalization, dout)
    if ndims(dout) != 2
        N, C, H, W = size(dout)
        dout = reshape(dout, N, :)
    end
    dx = __backward(self, dout)

    dx = reshape(dx, self.input_shape)
    return dx
end

function __backward(self::BatchNormalization, dout)
    dbeta = sum(dout, dims=1)
    dgamma = sum(self.xn .* dout, dims=1)
    dxn = self.gamma .* dout
    dxc = dxn ./ self.std
    dstd = -sum((dxn .* self.xc) ./ (self.std.^2), dims=1)
    dvar = 0.5 * dstd ./ self.std
    dxc .+= (2.0 / self.batch_size) * self.xc .* dvar
    dmu = sum(dxc, dims=1)
    dx = dxc .- dmu ./ self.batch_size
    
    self.dgamma = dgamma
    self.dbeta = dbeta
    
    return dx
end


mutable struct Convolution{T <: Real, Y <: Real}
    W::Array{T, 4} 
    b::Array{T, 2}
    stride::Integer
    pad::Integer
    # 中間データ（backward時に使用）
    x::Array{Y, 4}
    col::Array{T, 2}
    col_W::Array{T, 2}
    # 重み・バイアスパラメータの勾配
    dW::Array{T, 4}
    db::Array{T, 2}
end

function Convolution(W, b; stride=1, pad=0)
    t = eltype(W)
    if ndims(b)==1
        b = reshape(b, 1, :)
    end
    return Convolution(W, b, stride, pad, zeros(t, 0,0,0,0), zeros(t,0,0), zeros(t,0,0), zero(W), zero(b))
end

function forward(self::Convolution, x)
    FN, C, FH, FW = size(self.W)
    N, C, H, W = size(x)
    out_h = fld1(H + 2*self.pad - FH, self.stride)
    out_w = fld1(W + 2*self.pad - FW, self.stride)

    col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)
    col_W = reshape(self.W, (FN, :))'

    out = col * col_W .+ self.b
    out = permutedims(reshape(out, (N, out_h, out_w, :)), [1, 4, 2, 3])

    self.x = x
    self.col = col
    self.col_W = col_W

    return out
end

function backward(self::Convolution, dout)
    FN, C, FH, FW = size(self.W)
    dout = reshape(permutedims(dout, [1,3,4,2]), (:, FN))

    self.db = sum(dout, dims=1)
    dW = self.col' * dout
    self.dW = reshape(permutedims(dW, [2, 1]), (FN, C, FH, FW))

    dcol = dout * self.col_W'
    dx = col2im(dcol, size(self.x), FH, FW, stride=self.stride, pad=self.pad)

    return dx
end


mutable struct Pooling{T <: Real}
    pool_h::Integer
    pool_w::Integer
    stride::Integer
    pad::Integer
    x::Array{T, 4}
    arg_max
end

function Pooling(pool_h, pool_w; stride=2, pad=0)
    return Pooling(pool_h, pool_w, stride, pad, zeros(0,0,0,0), nothing)
end

function forward(self::Pooling, x)
    N, C, H, W = size(x)
    out_h = fld1(H - self.pool_h, self.stride)
    out_w = fld1(W - self.pool_w, self.stride)

    col = im2col(x, self.pool_h, self.pool_w, stride=self.stride, pad=self.pad)
    col = reshape(col, (:, self.pool_h*self.pool_w))

    arg_max = argmax(col, dims=2)
    out = maximum(col, dims=2)
    out = permutedims(reshape(out, (N, out_h, out_w, C)), (1, 4, 2, 3))

    self.x = x
    self.arg_max = arg_max

    return out
end

function backward(self::Pooling, dout)
    dout = permutedims(dout, (1, 3, 4, 2))
    
    pool_size = self.pool_h * self.pool_w
    dmax = zeros(length(dout), pool_size)
    dmax[self.arg_max] = reshape(dout, :)
    dmax = reshape(dmax, size(dout)..., pool_size) 
    
    dcol = reshape(dmax, size(dmax, 1) * size(dmax, 2) * size(dmax, 3), :)
    dx = col2im(dcol, size(self.x), self.pool_h, self.pool_w, stride=self.stride, pad=self.pad)
    
    return dx
end

end # module Layers