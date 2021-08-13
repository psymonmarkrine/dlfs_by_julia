import OrderedCollections: OrderedDict

using Plots
include("deep_convnet.jl") # DeepConvNet
include("../dataset/mnist.jl") # load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=false)

network = DeepConvNet()
load_params(network, "deep_convnet_params.h5")

println("calculating test accuracy ... ")
#sampled = 1000
#x_test = x_test[:sampled, :, :, :]
#t_test = t_test[:sampled]

classified_ids = zeros(Integer, 0)

acc = 0.0
batch_size = 1000

for i = 1:Integer(floor(size(x_test, 1) / batch_size))
    tx = x_test[(i-1)*batch_size+1:i*batch_size,:,:,:]
    tt = t_test[(i-1)*batch_size+1:i*batch_size]
    y = predict(network, tx, false)
    y = [i[2] for i=argmax(y, dims=2)]
    global classified_ids = vcat(classified_ids, y[:])
    global acc += sum(y .== eltype(y).(tt))
end

acc = acc / size(x_test, 1)
println("test accuracy : $acc")

nx, ny = (4, 5)
max_view = nx*ny
current_view = 1

println("======= misclassified result =======")
println("{view index => (label, inference), ...}")

# mis_pairs = OrderedDict()
p = []
for (i, val) = enumerate(classified_ids .== eltype(classified_ids).(t_test))
    if !val
        plot(xticks=false, yticks=false)
        push!(p, heatmap!(x_test[i,1,:,end:-1:1]', color=:grays, cbar=false))
        # mis_pairs[current_view] = (t_test[i]-1, classified_ids[i]-1)
        println("$current_view => ($(t_test[i]-1), $(classified_ids[i]-1))")
        global current_view += 1
        if current_view > max_view
            break
        end
    end
end
plot(p..., layout=(nx, ny))
# savefig("../image/ch08/fig.png")
