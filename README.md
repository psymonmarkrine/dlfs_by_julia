# ゼロから作るDeep LearningをJuliaで学びたかった

自分用まとめ  
章節に沿いながら大雑把にまとめてコードをJuliaに書き換えていく  

書籍  
ゼロから学ぶDeep Learning ～Pythonで学ぶディープラーニングの理論と実装～  
> https://www.oreilly.co.jp/books/9784873117584/  
> https://github.com/oreilly-japan/deep-learning-from-scratch  

***

[まえがき](./md/preface.md)

## 目次

### 1章 Julia入門
1.1 [Julia とは](./md/ch01.md#11-juliaとは)  
1.2 [Juliaのインストール](./md/ch01.md#12-juliaのインストール)   
　1.2.1 [Juliaのバージョン](./md/ch01.md#121-juliaのバージョン)  
　1.2.2 [使用する外部ライブラリ](./md/ch01.md#122-使用する外部ライブラリ)  
1.3 [Juliaインタプリンタ](./md/ch01.md#13-juliaインタプリンタ)  
　1.3.1 [算術計算](./md/ch01.md#131-算術計算)  
　1.3.2 [データ型](./md/ch01.md#132-データ型)  
　1.3.3 [変数](./md/ch01.md#133-変数)  
　1.3.4 [配列の生成](./md/ch01.md#134-配列の生成)  
　1.3.5 [N次元配列](./md/ch01.md#135-n次元配列)  
　1.3.6 [配列の算術演算](./md/ch01.md#136-配列の算術演算)  
　1.3.7 [ブロードキャスト](./md/ch01.md#137-ブロードキャスト)  
　1.3.8 [ディクショナリ](./md/ch01.md#138-ディクショナリ)  
　1.3.9 [ブーリアン](./md/ch01.md#139-ブーリアン)  
　1.3.10 [if文](./md/ch01.md#1310-if文)  
　1.3.11 [for文](./md/ch01.md#1311-for文)  
　1.3.12 [関数](./md/ch01.md#1312-関数)  
1.4 [Juliaスクリプトファイル](./md/ch01.md#14-juliaスクリプトファイル)  
1.5 [Plots](./md/ch01.md#15-plots)  
　1.5.1 [Plotsのインポート](./md/ch01.md#151-plotsのインポート)  
　1.5.2 [単純なグラフ描画](./md/ch01.md#152-単純なグラフ描画)  
　1.5.3 [Plotsの機能](./md/ch01.md#153-plotsの機能)  

### 2章 パーセプトロン
2.1 パーセプトロンとは   
2.2 単純な論理回路  
　2.2.1 ANDゲート  
　2.2.2 NANDゲートとORゲート  
2.3 パーセプトロンの実装  
　2.3.1 [簡単な実装](./md/ch02.md#231-簡単な実装)  
　2.3.2 [重みとバイアスの導入](./md/ch02.md#232-重みとバイアスの導入)  
　2.3.3 [重みとバイアスによる実装](./md/ch02.md#233-重みとバイアスによる実装)  
2.4 パーセプトロンの限界  
　2.4.1 XORゲート  
　2.4.2 線形と非線形  
2.5 多層パーセプトロン  
　2.5.1 既存ゲートの組み合わせ  
　2.5.2 [XORゲートの実装](./md/ch02.md#252-xorゲートの実装)  
2.6 NANDからコンピュータへ  
2.7 まとめ

### 3章 ニューラルネットワーク

3.1 パーセプトロンからニューラルネットワークへ  
　3.1.1 ニューラルネットワークの例  
　3.1.2 パーセプトロンの復習  
　3.1.3 活性化関数の登場  
3.2 活性化関数  
　3.2.1 シグモイド関数  
　[3.2.2 ステップ関数の実装](./md/ch03.md#322-ステップ関数の実装)  
　[3.2.3 ステップ関数のグラフ](./md/ch03.md#323-ステップ関数のグラフ)  
　[3.2.4 シグモイド関数の実装](./md/ch03.md#324-シグモイド関数の実装)  
　3.2.5 シグモイド関数とステップ関数の比較
　3.2.6 非線形関数
　[3.2.7 ReLU関数](./md/ch03.md#327-ReLU関数)  
3.3 多次元配列の計算  
　[3.3.1 多次元配列](./md/ch03.md#331-多次元配列)  
　[3.3.2 行列の積](./md/ch03.md#332-行列の積)  
　[3.3.3 ニューラルネットワークの行列の積](./md/ch03.md#333-ニューラルネットワークの行列の積)  
3.4 3層ニューラルネットワーク  
　3.4.1 記号の確認  
　[3.4.2 各層における信号伝達の実装](./md/ch03.md#342-各層における信号伝達の実装)
　[3.4.3 実装のまとめ](./md/ch03.md#343-実装のまとめ)

### 4章 ニューラルネットワークの学習

### 5章 誤差逆伝搬法

### 6章 学習に関するテクニック

### 7章 ニューラルネットワーク

### 8章 ディープラーニング

### 
