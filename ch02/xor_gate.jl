include("and_gate.jl")
include("or_gate.jl")
include("nand_gate.jl")


function XOR(x1, x2)
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
end

if abspath(PROGRAM_FILE) == @__FILE__
    for xs = [(0, 0), (1, 0), (0, 1), (1, 1)]
        y = XOR(xs[1], xs[2])
        println("$xs -> $y")
    end
end