include("layer_naive.jl")

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = forward(mul_apple_layer, apple, apple_num)
price = forward(mul_tax_layer, apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = backward(mul_tax_layer, dprice)
dapple, dapple_num = backward(mul_apple_layer, dapple_price)

println("price:", Int(round(price)))
println("dApple:", dapple)
println("dApple_num:", Int(round(dapple_num)))
println("dTax:", dtax)
