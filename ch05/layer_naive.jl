mutable struct MulLayer
    x
    y
end

function MulLayer()
    return MulLayer(nothing, nothing)
end

function forward(self::MulLayer, x, y)
    self.x = x
    self.y = y                
    out = x * y

    return out
end

function backward(self::MulLayer, dout)
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy
end


struct AddLayer end

function forward(self::AddLayer, x, y)
    out = x + y

    return out
end

function backward(self::AddLayer, dout)
    dx = dout * 1
    dy = dout * 1

    return dx, dy
end
