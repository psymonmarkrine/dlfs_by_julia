""" 
   MNIST_DL

MNISTデータセットをダウンロードするためのモジュール

`.gzファイル`をダウンロードして`.h5ファイル`へと変換する。
`.gzファイル`を読むために追加パッケージ `GZip` が必要だが、
リポジトリに`.h5ファイル`を同梱しているためこのモジュールを使用する必要はないはず。
"""
module MNIST_DL
    import Downloads
    import GZip, HDF5

    url_base = "http://yann.lecun.com/exdb/mnist/"
    key_file = Dict(
        "train_img"=>"train-images-idx3-ubyte.gz",
        "train_label"=>"train-labels-idx1-ubyte.gz",
        "test_img"=>"t10k-images-idx3-ubyte.gz",
        "test_label"=>"t10k-labels-idx1-ubyte.gz"
    )

    dataset_dir = dirname(abspath(@__FILE__))
    save_file = joinpath(dataset_dir, "mnist.h5")

    img_size = 784


    function _download(file_name)
        file_path = joinpath(dataset_dir, file_name)

        if isfile(file_path)
            return
        end

        println("Downloading $file_name ... ")
        headers = Dict("User-Agent" => "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0")
        Downloads.download(url_base*file_name, file_path, headers=headers)
        println("Done")
    end

    function download_mnist()
        for (k, v) in key_file
        _download(v)
        end
    end

    function _load_label(file_name)
        file_path = joinpath(dataset_dir, file_name)

        println("Converting $file_name to Array ...")
        labels = GZip.open(file_path, "r") do f
            read(f)
        end
        print("$(size(labels)) ")
        println("Done")

        return labels[9:end] .+ 1
    end

    function _load_img(file_name)
        file_path = joinpath(dataset_dir, file_name)

        println("Converting $file_name to Array ...")
        data = GZip.open(file_path, "r") do f
            read(f)
        end
        print("$(size(data)) ")
        data = collect(reshape(data[17:end], (img_size, :))')
        println("Done")

        return data
    end

    function _convert_array()
        dataset = Dict()
        dataset["train_img"] =  _load_img(key_file["train_img"])
        dataset["train_label"] = _load_label(key_file["train_label"])
        dataset["test_img"] = _load_img(key_file["test_img"])
        dataset["test_label"] = _load_label(key_file["test_label"])

        return dataset
    end

    function init_mnist()
        download_mnist()
        dataset = _convert_array()
        println("Creating HDF5 file ...")
        for (k, v) in dataset
            HDF5.h5write(save_file, k, v)
        end
        println("Done!")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    Main.MNIST_DL.init_mnist()
end