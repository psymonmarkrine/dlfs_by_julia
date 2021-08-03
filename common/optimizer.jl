mutable struct SGD # """確率的勾配降下法（Stochastic Gradient Descent）"""
    lr::AbstractFloat
end

SGD() = SGD(0.01)

function update(self::SGD, params, grads)
    for (key,_) = params
        params[key] .-= self.lr * grads[key]
    end
end


mutable struct Momentum # """Momentum SGD"""
    lr::AbstractFloat
    momentum::AbstractFloat
    v::IdDict
end
 Momentum(lr=0.01, momentum=0.9) = Momentum(lr, momentum, IdDict())
         
function update(self::Momentum, params, grads)
    for (key, val) = params
        get!(self.v, key, zero(val))
        self.v[key] = self.momentum .* self.v[key] .- self.lr * grads[key] 
        params[key] .+= self.v[key]
    end
end


mutable struct Nesterov # """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""
    lr::AbstractFloat
    momentum::AbstractFloat
    v::IdDict
end

Nesterov(lr=0.01, momentum=0.9) = Nesterov(lr, momoentum, IdDict())

function update(self::Nesterov, params, grads)
    for (key, val) = params
        get!(self.v, key, zero(val))
        params[key] .+= self.momentum * self.momentum * self.v[key]
        params[key] .-= (1 + self.momentum) * self.lr * grads[key]
        self.v[key] .*= self.momentum
        self.v[key] .-= self.lr * grads[key]
    end
end


mutable struct AdaGrad # """AdaGrad"""
    lr::AbstractFloat
    h::IdDict
end
AdaGrad(lr=0.01) = AdaGrad(lr, IdDict())

function update(self::AdaGrad, params, grads)
    for (key,val) = params
        get!(self.h, key, zero(val))
        self.h[key] .+= grads[key].^2
        params[key] .-= self.lr * grads[key] ./ (sqrt.(self.h[key]) .+ 1e-7)
    end
end


mutable struct RMSprop # """RMSprop"""
    lr::AbstractFloat
    decay_rate::AbstractFloat
    h::IdDict
end

RMSprop(lr=0.01, decay_rate = 0.99) = RMSprop(lr, decay_rate, IdDict())

function update(self::RMSprop, params, grads)
    for (key,val) = params
        get!(self.h, key, zero(val))
        self.h[key] .*= self.decay_rate
        self.h[key] .+= (1 - self.decay_rate) * grads[key].^2
        params[key] .-= self.lr * grads[key] ./ (sqrt.(self.h[key]) .+ 1e-7)
    end
end


mutable struct Adam # """Adam (http://arxiv.org/abs/1412.6980v8)"""
    lr::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    iter::Integer
    m::IdDict
    v::IdDict
end
Adam(lr=0.001, beta1=0.9, beta2=0.999) = Adam(lr, beta1, beta2, 0, IdDict(), IdDict())

function update(self::Adam, params, grads)
    self.iter += 1
    lr_t  = self.lr * sqrt(1.0 - self.beta2 ^ self.iter) / (1.0 - self.beta1 ^ self.iter)
    
    for (key, val) = params
        get!(self.m, key, zero(val))
        get!(self.v, key, zero(val))

        self.m[key] .+= (1 - self.beta1) * (grads[key] - self.m[key])
        self.v[key] .+= (1 - self.beta2) * (grads[key].^2 - self.v[key])
        
        params[key] .-= lr_t * self.m[key] ./ (sqrt.(self.v[key]) .+ 1e-7)
        
        #unbias_m .+= (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
        #unbisa_b .+= (1 - self.beta2) * (grads[key] .* grads[key] - self.v[key]) # correct bias
        #params[key] .+= self.lr * unbias_m ./ (sqrt.(unbisa_b) + 1e-7)
    end
end
