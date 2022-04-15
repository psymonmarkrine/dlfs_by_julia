include("mnist_dl.jl")

module MNIST
    import MNIST_DL: init_mnist, key_file, train_num, test_num, img_size
    import Downloads
    import HDF5

    dataset_dir = dirname(abspath(@__FILE__))
    save_file = joinpath(dataset_dir, "mnist.h5")

    img_dim = (1, 28, 28)
    

    function _change_one_hot_label(X)
        T = zeros(Float32, size(X, 1), 10)
        for (idx, row) in enumerate(X)
            T[idx, row] = 1
        end

        return T
    end


    function load_mnist(; normalize=true, flatten=true, one_hot_label=false)
        """MNISTデータセットの読み込み
        Parameters
        ----------
        normalize : 画像のピクセル値を0.0~1.0に正規化する
        one_hot_label :
            one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
            one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
        flatten : 画像を一次元配列に平にするかどうか
        Returns
        -------
        (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
        """
        if !isfile(save_file)
            include(joinpath(dirname(save_file),"mnist_dl.jl"))
            init_mnist()
        end

        dataset = Dict()
        for (k, v) in key_file
            dataset[k] = HDF5.h5read(save_file, k)
        end

        if normalize
            for key in ("train_img", "test_img")
                dataset[key] = Float32.(dataset[key])
                dataset[key] ./= 255.0
            end
        end

        if one_hot_label
            dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
            dataset["test_label"] = _change_one_hot_label(dataset["test_label"])
        end

        if !flatten
            for key in ("train_img", "test_img")
                dataset[key] = reshape(dataset[key], :, 1, 28, 28)
            end
        end


        return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])
    end
end
