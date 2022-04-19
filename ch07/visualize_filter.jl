include("simple_convnet.jl")

using Plots
using .SimpleConvNet_ch07

function filter_show(filters, nx=5)
    FN, C, FH, FW = size(filters)
    ny = ceil(Int, FN / nx)
 
    p = []
    for i = 1:FN
        plot(xticks=nothing, yticks=nothing)
        push!(p, heatmap!(filters[i, 1, :, :], color=:grays, cbar=false))
    end
    plot(p..., layout=(nx, ny))
end

network = SimpleConvNet()
# ランダム初期化後の重み
filter_show(network.params["W1"])
savefig("../image/ch07/fig07-24a.png")

# 学習後の重み
load_params(network, "params.h5")
filter_show(network.params["W1"])
savefig("../image/ch07/fig07-24b.png")
