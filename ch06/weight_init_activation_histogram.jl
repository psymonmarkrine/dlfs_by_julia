import OrderedCollections: OrderedDict
using Plots


function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function ReLU(x)
    return max(0, x)
end

# function tanh(x)
#     return tanh(x)
# end
    
input_data = randn(1000, 100)  # 1000個のデータ
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層
activations = OrderedDict()  # ここにアクティベーションの結果を格納する

for i = 1:hidden_layer_size
    if i != 1
        x = activations[i-1]
    else
        x = input_data
    end

    # 初期値の値をいろいろ変えて実験しよう！
    w = randn(node_num, node_num) * 1
    # w = randn(node_num, node_num) * 0.01
    # w = randn(node_num, node_num) * sqrt(1.0 / node_num)
    # w = randn(node_num, node_num) * sqrt(2.0 / node_num)


    a = x * w


    # 活性化関数の種類も変えて実験しよう！
    # z = sigmoid.(a)
    z = ReLU.(a)
    # z = tanh.(a)

    activations[i] = z
end

# ヒストグラムを描画
p = []
for (i, a) = activations
    if i == 1
        plot()
    else
        plot(yticks=nothing)
    end
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    
    push!(p, histogram!(a[:], bins=30, title="$i-layer", leg=false))
end
plot(p..., layout=(1,length(activations)), size=(800,200))
savefig("../image/ch06/fig06-10.png")
