using Plots

function sigmoid(x::Real)
    return 1 / (1 + exp(-x))    
end

# function step_function(x)
#     return typeof(x)(x > 0)
# end

function step_function(x::T) where T <: Real
    return T(x > 0)
end

x = range(-5.0, 5.0, step=0.1)
y1 = sigmoid.(x)
y2 = step_function.(x)
plot(ylims = (-0.1, 1.1)) # 図で描画するy軸の範囲を指定
plot!(x, y1, leg=false)
plot!(x, y2, leg=false, linestyle = :dash)
savefig("../image/ch03/fig03-08.png")