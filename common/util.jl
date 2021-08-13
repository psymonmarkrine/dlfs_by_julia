import Random: randperm

function smooth_curve(x)
    """損失関数のグラフを滑らかにするために用いる
    雑ハミング窓
    """
    window_len = 11
    w = 0.6 .- 0.4cos.(2π .* (0:window_len-1)./(window_len-1))
    w ./= sum(w)
    y = [w' * x[i:(i+window_len-1)] for i=1:(size(x,1)-window_len+1)]
    
    return y
end

function shuffle_dataset(x, t)
    """データセットのシャッフルを行う
    Parameters
    ----------
    x : 訓練データ
    t : 教師データ
    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = randperm(size(x, 1))
    x = ndims(x) == 2 ? x[permutation,:] : x[permutation,:,:,:]
    t = ndims(t) == 1 ? t[permutation] : t[permutation,:]

    return x, t
end

function conv_output_size(input_size::Integer, filter_size::Integer, stride::Integer=1, pad::Integer=0)
    return (input_size + 2pad - filter_size) / stride + 1
end

function padzero(x::Array{T}, v::Vector{Tuple{Y,Y}}) where {T <: Real, Y <: Integer}
    """0パディングするための関数
    """
    n = min(ndims(x), length(v))
    t = eltype(x)

    for i=1:n
        s = size(x)
        s1 = [j for j=s]
        s2 = [j for j=s]
        s1[i] = v[i][1]
        s2[i] = v[i][2]
        pad1 = zeros(t, Tuple(s1))
        pad2 = zeros(t, Tuple(s2))
        x = cat(pad1, x, pad2, dims=i)
    end

    return x
end

function im2col(input_data::Array{T, 4}, filter_h::Integer, filter_w::Integer; stride::Integer=1, pad::Integer=0) where T <: Real
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = size(input_data)
    out_h = div(H + 2pad - filter_h, stride) + 1
    out_w = div(W + 2pad - filter_w, stride) + 1

    img = padzero(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)])
    col = zeros(N, C, filter_h, filter_w, out_h, out_w)

    for y = 1:filter_h
        y_max = y + stride*(out_h-1)
        for x = 1:filter_w
            x_max = x + stride*(out_w-1)
            col[:, :, y, x, :, :] = img[:, :, y:stride:y_max, x:stride:x_max]
        end
    end

    col = reshape(permutedims(col, [1, 5, 6, 2, 3, 4]), N*out_h*out_w, :)
    return col
end

function col2im(col, input_shape, filter_h::Integer, filter_w::Integer; stride::Integer=1, pad::Integer=0)
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = div(H + 2*pad - filter_h, stride) + 1
    out_w = div(W + 2*pad - filter_w, stride) + 1
    col = permutedims(reshape(col, (N, out_h, out_w, C, filter_h, filter_w)), [1, 4, 5, 6, 2, 3])

    img = zeros(N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1)
    for y = 1:filter_h
        y_max = y + stride*(out_h-1)
        for x = 1:filter_w
            x_max = x + stride*(out_w-1)
            img[:, :, y:stride:y_max, x:stride:x_max] .+= col[:, :, y, x, :, :]
        end
    end

    return img[:, :, (1:H) .+ pad, (1:W) .+ pad]
end

function weight_init_randn(shape, init_std=0.01)
    input_size = length(shape)==2 ? shape[1] : *(shape[2:end]...)
    scale = init_std
    if lowercase("$init_std") in ("relu", "he")
        scale = sqrt(2.0 / input_size)  # ReLUを使う場合に推奨される初期値
    elseif lowercase("$init_std") in ("sigmoid", "xavier")
        scale = sqrt(1.0 / input_size)  # sigmoidを使う場合に推奨される初期値
    end

    return scale * randn(shape)
end
