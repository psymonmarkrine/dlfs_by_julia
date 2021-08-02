using Plots
include("optimizer.jl")


function f(x, y)
    return x^2 / 20.0 + y^2
end


function df(x, y)
    return x / 10.0, 2y
end

init_pos = (-7.0, 2.0)
params = Dict(
    "x" => init_pos[1],
    "y" => init_pos[2]
)
grads = Dict(
    "x" => 0.0,
    "y" => 0.0
)


optimizers = Dict(
    "SGD" => SGD(0.95),
    "Momentum" => Momentum(0.1),
    "AdaGrad" => AdaGrad(1.5),
    "Adam" => Adam(0.3)
)
# idx = 1
p = []

for (key, val) = optimizers
    optimizer = val
    x_history = zeros(0)
    y_history = zeros(0)
    params["x"], params["y"] = init_pos[1], init_pos[2]
    
    for i = 1:30
        append!(x_history, params["x"])
        append!(y_history, params["y"])
        
        grads["x"], grads["y"] = df(params["x"], params["y"])
        update(optimizer, params, grads)
    end

    x = -10:0.01:10
    y = -5:0.01:5
    
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    Z = f.(X, Y)
    
    # for simple contour line  
    mask = Z .> 7
    Z[mask] .= 0
    
    # plot 
    # subplot(2, 2, idx)
    # idx += 1
    plot(0, 0, marker=:+, xlabel="x", ylabel="y", xlim=(-10,10), ylim=(-10,10), leg=false)
    contour!(x, y, Z)
    push!(p, plot!(x_history, y_history, title=key, marker=:c, color=:red, leg=false))
    # colorbar()
    # spring()
end
plot(p..., layout=(2,2), size=(1200,800))
savefig("../image/ch06/fig06-08.png")
