using Plot 

# データの作成
x = range(0, 6, step=0.1)
y1 = sin.(x)
y2 = cos.(x)

# グラフの描画
plot(xlabel="x", ylabel="y", title="sin & cos") # x軸、y軸のラベルとタイトルの設定
plot!(x, y1, label="sin")
plot!(x, y2, label="cos", linestyle=:dash)
