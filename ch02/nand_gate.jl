function NAND(x1, x2)
    x = [x1, x2]
    w = [-0.5, -0.5]
    b = 0.7
    tmp = w'*x + b
    if tmp <= 0
        return 0
    else
        return 1
    end
end

for xs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    y = NAND(xs[1], xs[2])
    println("$xs -> $y")
end
