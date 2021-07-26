using Plots
include("gradient_2d.jl") # numerical_gradient


function gradient_descent(f, init_x; lr=0.01, step_num=100)
    x = init_x
    x_history = zeros(step_num,length(x))

    for i = 1:step_num
        x_history[i,:] = x

        grad = numerical_gradient(f, x)
        x .-= lr * grad
    end

    return x, x_history
end

function function_2(x)
    return x[0]^2 + x[1]^2
end

init_x = [-3.0, 4.0]

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plot(xlim=[-3.5, 3.5], ylim=[-4.5, 4.5], xlabel="X0", ylabel="X1", leg=false)
scatter!(x_history[:,1], x_history[:,2])
plot!( [-5, 5], [0,0], linestyle=:dash, color=:blue)
plot!( [0,0], [-5, 5], linestyle=:dash, color=:blue)