TensorFlowではじめるDeepLearning実践入門　サンプルコード
====

このプログラムはインプレスが発行する書籍「[TensorFlowではじめるDeepLearning実践入門](https://book.impress.co.jp/books/1117101113)」におけるサンプルコードです。

# 構成

本プログラムのフォルダ構成は以下のようになっています。

| フォルダ名 | 説明 |
---|---
| graph | 計算グラフの解説用ソースコード類。 |
| nn | 全結合ニューラルネットワークによる手書き数字認識のソースコード類。 |
| board | TensorBoardの解説用ソースコード類。 |
| conv | 畳み込みニューラルネットワークによる手書き数字認識のソースコード類。 |
| save_model | モデル保存の解説用ソースコード類。|
| s_mnist | RNNによる手書き数字認識のソースコード類。 |
| word2vec | word2vecの解説用ソースコード類。 |
| tfrecord | TFRecordの読み書きの解説用ソースコード類。 |
| nic | イメージキャプショニング用のソースコード類。 |

# 必要環境
書籍に記載済み。後ほどrequirement.txtを作成予定。

# ライセンス
このプログラムは[MITライセンス](https://opensource.org/licenses/mit-license.php)です。

# 正誤表

- 124ページ中段数式　（誤）z * -log(x) + (1-z) * -log(1-x') -> (正) z * -log(x) + (1-z) * -log(1-x)

# メモ

- ch03    

https://www.cs.toronto.edu/~kriz/cifar.html
https://gist.github.com/katsugeneration/6fa6bdafb36cc667caea6c54cc7bfed6




