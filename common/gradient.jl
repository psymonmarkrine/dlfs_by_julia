module Gradient

function _numerical_gradient_1d(f, x::Vector{T}) where T <: AbstractFloat
    h = 1e-4 # 0.0001
    grad = zero(x)
    
    for idx=1:size(x,1)
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / 2h
        
        x[idx] = tmp_val # 値を元に戻す
    end
    return grad
end

function numerical_gradient_2d(f, X::Vector{T}) where T <: AbstractFloat
    return _numerical_gradient_1d(f, X)
end

function numerical_gradient_2d(f, X::Matrix{T}) where T <: AbstractFloat
    grad = zero(X)
    for idx=1:size(X,1)
        grad[idx,:] = _numerical_gradient_1d(f, X[idx,:])
    end
    return grad
end


function numerical_gradient(f, x)
    h = 1e-4 # 0.0001
    grad = zero(x)
    
    for idx=CartesianIndices(x)
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / 2h
        
        x[idx] = tmp_val # 値を元に戻す
    end
    
    return grad
end

end # module Gradient