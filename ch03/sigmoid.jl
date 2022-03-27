using Plots

function sigmoid(x::Real)
    return 1 / (1 + exp(-x))
end

X = range(-5.0, 5.0, step=0.1)
Y = sigmoid.(X)
plot(X, Y, leg = false)
plot!(ylims = (-0.1, 1.1))  # 図で描画するy軸の範囲を指定
savefig("../image/ch03/fig03-07.png")