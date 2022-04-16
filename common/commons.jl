include("functions.jl")
include("gradient.jl")
include("util.jl")
include("optimizer.jl")
include("layers.jl")
include("trainer.jl")

module MultiLayerNets

export  MultiLayerNet, MultiLayerNetExtend,
        predict, loss, accuracy, numerical_gradient, gradient

import  OrderedCollections: OrderedDict

using   Layers
import  Gradient: numerical_gradient

# include("multi_layer_net.jl")
# include("multi_layer_net_extend.jl")

end
