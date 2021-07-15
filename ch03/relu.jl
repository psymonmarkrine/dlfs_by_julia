using Plots

function relu(x)
    return max(0, x)
end

x = range(-5.0, 5.0, step=0.1)
y = relu.(x)
plot(x, y, leg = false)
plot!(ylims = (-1.0, 5.5))
savefig("../image/ch03/fig03-09.png")
