using Plots

# function step_function(x)
#     return typeof(x)(x > 0)
# end

function step_function(x::T) where T <: Real
    return T(x > 0)
end

X = range(-5.0, 5.0, step=0.1)
Y = step_function.(X)
plot(X, Y, leg = false)
plot!(ylims = (-0.1, 1.1))  # 図で描画するy軸の範囲を指定
savefig("../image/ch03/fig03-06.png")
