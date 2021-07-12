using Plot 

# データの作成
x = range(0, 6, step=0.1)
y = sin.(x)

# グラフの描画
plot(x, y)
