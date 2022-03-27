using Plots


function _numerical_gradient_no_batch(f, x)
    h = 1e-4  # 0.0001
    grad = zero(x)
    
    for idx = 1:size(x,1)
        tmp_val = x[idx]
        x[idx] += h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] -= 2h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / 2h
        
        x[idx] = tmp_val  # 値を元に戻す
    end
    return grad
end

function numerical_gradient(f, X::Vector{T}) where T
    return _numerical_gradient_no_batch(f, X)
end

function numerical_gradient(f, X::Matrix{T}) where T
    grad = zero(X)
    
    for idx=1:size(X,1)
        grad[idx,:] = _numerical_gradient_no_batch(f, X[idx,:])
    end
    return grad
end

function function_2(x::Vector{T}) where T <: Real
    return sum(x.^2)
end

function function_2(x::Matrix{T}) where T <: Real
    return function_2.([x[i,:] for i=1:size(x,1)])
end

function tangent_line(f, x)
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) .- d.*x
    return (t)->(d.*t .+ y)
end


if abspath(PROGRAM_FILE) == @__FILE__
    ratio = 0.08
    x0 = -2:0.25:2.5
    x1 = -2:0.25:2.5
    X = repeat(x0, outer=length(x1))
    Y = repeat(x1, inner=length(x0))

    grad = ratio*numerical_gradient(function_2, [X Y])

    quiver(X, Y, quiver=(-grad[:,1],  -grad[:,2]))
    plot!(xlim=[-2,2], ylim=[-2,2], xlabel="x0", ylabel="x1")
    savefig("../image/ch04/fig04-09.png")
end