include("layer_naive.jl")

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = forward(mul_apple_layer, apple, apple_num)  # (1)
orange_price = forward(mul_orange_layer, orange, orange_num)  # (2)
all_price = forward(add_apple_orange_layer, apple_price, orange_price)  # (3)
price = forward(mul_tax_layer, all_price, tax)  # (4)

# backward
dprice = 1
dall_price, dtax = backward(mul_tax_layer, dprice)  # (4)
dapple_price, dorange_price = backward(add_apple_orange_layer, dall_price)  # (3)
dorange, dorange_num = backward(mul_orange_layer, dorange_price)  # (2)
dapple, dapple_num = backward(mul_apple_layer, dapple_price)  # (1)

println("price:", round(Int, price)
println("dApple:", dapple)
println("dApple_num:", round(Int, dapple_num)
println("dOrange:", dorange)
println("dOrange_num:", round(Int, dorange_num)
println("dTax:", dtax)
