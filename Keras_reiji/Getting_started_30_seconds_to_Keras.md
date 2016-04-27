## Getting started: 30 seconds to Keras
Kerasはmodelでデータ構造を定義する.
その中で最もよく使われるのがSequentialで,より複雑なアーキテクチャを持つレイヤーを定義しようと思ったら[ Keras function API ](http://keras.io/getting-started/functional-api-guide)を確認する必要がある.
Sequential modelの使用は以下のように行う.

```
	from keras.model import Sequential
	model = Sequential()
```

層を積み重ねる場合は`.add()`を使う.

```
from keras.layers.core import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```

モデルの作成が終わったら`.compile()`を使ってモデルの学習プロセスの設定を行う.
```
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

もしオプティマイザをさらに詳しく記述する必要があるのなら,それも可能である.
Kerasは合理的でシンプルに問題を対処することを信条としているが,その一方でユーザのカスタマイズはどの部分でも行えるようになっている.

```
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

ここまで計算路を確定したらトレーニングデータを回すことができる.
```
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

手動でバッチを作成したモデルに食わせることもできる.
```
model.train_on_batch(X_batch, Y_batch)
```

作成したモデルの評価を一行で行うことができる.
```
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
```
新しいデータに対して予測を行う場合は以下のように行う.
```
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```
一方向の単純なdeep learningはこのように簡単に記述できる.
Deep learningの考えはとてもシンプルなのになぜ実装に苦痛を伴わなければならないのか（いやそんなことはない）.

より詳しいチュートリアルは
+ [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide)
+ [Getting started with the function API](http://keras.io/getting-started/functional-api-guide)

で確認することができる.  
githubのexampleページにはLSTMなどのモデルが置いてある.
