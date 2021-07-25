using Plots

function numerical_diff(f, x)
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
end

function function_1(x)
    return 0.01x^2 + 0.1x 
end

function tangent_line(f, x)
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return (t)->(d*t + y)
end

x = 0:0.1:20
y = function_1.(x)

tf = tangent_line(function_1, 5)
y2 = tf.(x)

plot(xlabel="x", ylabel="f(x)", leg=false)
plot!(x, y)
plot!(x, y2)
savefig("../image/ch04/fig04-06.png")
