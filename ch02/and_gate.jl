function AND(x1, x2)
    x = [x1, x2]
    w = [0.5, 0.5]
    b = -0.7
    tmp = w'*x + b
    if tmp <= 0
        return 0
    else
        return 1
    end
end

# パーセプトロンからニューラルネットワークへと発展させるために
# ベクトルの内積を行うようにプログラムを書き換えているが、
# この程度の単純な計算の場合は以下の方が圧倒的に計算速度が早い
#
# function AND(x1, x2)
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = w1*x1 + w2*x2
#     if tmp <= theta
#         return 0
#     else
#         return 1
#     end
# end

if abspath(PROGRAM_FILE) == @__FILE__
    for xs = [(0, 0), (1, 0), (0, 1), (1, 1)]
        y = AND(xs[1], xs[2])
        println("$xs -> $y")
    end
end
