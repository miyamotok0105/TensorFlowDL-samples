


# go-lang

公式からおとすもよし    
https://golang.org/dl/    
goenvを入れるのもあり。    

goの環境設定    

```
go verion
go env GOROOT
//uninstall
//rm -rvf /usr/local/go/
go run helloworld.go
```

goの環境設定2    

```
TF_TYPE="cpu" # Change to "gpu" for GPU support
 TARGET_DIRECTORY='/usr/local'
 curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.8.0.tar.gz" |
 sudo tar -C $TARGET_DIRECTORY -xz
```

goの環境設定3    

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/lib
```

goのtf用ライブラリを入れる    


```
go get -u google.golang.org/grpc
go get github.com/tensorflow/tensorflow/tensorflow/go
go test github.com/tensorflow/tensorflow/tensorflow/go
go run hello_tf.go
```


プロトコルバッファ用    

```
brew install protobuf
go get -u github.com/golang/protobuf/protoc-gen-go
export GOPATH=$HOME/go
PATH=$PATH:$GOPATH/bin
```

# mnistデータ

```
git clone https://github.com/myleott/mnist_png.git
cd mnist_png
tar -zxvf mnist_png.tar.gz
```

# serving


```
#ここをちょっと変えてport9000が空いてる状態にしてほしい。
docker build --pull -t tensorflow-model-server -f Dockerfile .
docker run -it [コンテナid]
apt-get update
apt-get install python3-pip python3
apt-get install -y software-properties-common # if not already installed
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update
apt-get upgrade
apt-get dist-upgrade
pip3 install tensorflow
docker cp freee_mnist2.py [コンテナID]:/usr/local/src
docker cp mnist_input_data.py [コンテナID]:/usr/local/src
docker cp mnist_input_data.py [コンテナID]:/usr/local/src
```

サービングの登録まではうまくいく。    

```
python3 freee_mnist2.py --training_iteration=1000 --model_version=1 /usr/local/src/tmp2
tensorflow_model_server --port=9000 --model_name=tmp2 --model_base_path=/usr/local/src/tmp2
```

これは動いてない。


```
go run freee_mnist2_client.go --serving-address localhost:9000 mnist_png/mnist_png/testing/2/1174.png
```


# saved model

saved modelにして、それをサービングする。    

```
wget https://www.dropbox.com/s/wlqcmfwnny9cv43/Baskerville-01.png
```

saved model動かして見る    


```
python mnist_saved_model.py --training_iteration=1000 --model_version=1 tmp
```


# 書いてる通りに動かす

```
git clone --recursive https://github.com/tensorflow/serving.git
cd serving
git checkout r1.4
rm -d tensorflow
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout d752244fbaad5e4268243355046d30990f59418f
cd ..
cd ..
```

登録    

```
protoc -I=serving -I serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow_serving/apis/*.proto
protoc -I=serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow/tensorflow/core/framework/*.proto
protoc -I=serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow/tensorflow/core/protobuf/{saver,meta_graph}.proto
protoc -I=serving/tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow/tensorflow/core/example/*.proto
```

くそ。dockerのポートが開いてない。。    
多分動くからもぉ寝る。    


```
go run freee_mnist2_client.go --serving-address vibrant_zhukovsky:9000 mnist_png/mnist_png/testing/2/1174.png
```



# 参照

非常に参考になりました    
http://developers.freee.co.jp/entry/serve-ml-model-by-tensorflow-serving    
https://github.com/yonekawa/tensorflow-serving-playground    

https://github.com/tensorflow/serving/issues/819    


