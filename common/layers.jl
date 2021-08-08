import Statistics: mean

include("functions.jl")
include("util.jl") # im2col, col2im

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

#=
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
=#