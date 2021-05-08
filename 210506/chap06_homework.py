# -*- coding: utf-8 -*-
"""chap06_homework.ipynb

# 第4回講義 宿題

## 課題

CNNを用いて、FashionMNISTの高精度な分類器を実装してみましょう。
モデルのレイヤーを変更してみるなどして精度の向上にチャンレンジして下さい。 精度上位者はリーダーボードに載ります。

### 目標値

Accuracy 93%

### ルール

- 訓練データはx_train、 t_train、テストデータはx_testで与えられます。
- 予測ラベルは one_hot表現ではなく0~9のクラスラベル で表してください。
- **下のセルで指定されているx_train、t_train以外の学習データは使わないでください。**
- ただし、**torch.nn.Conv2dのような高レベルのAPIは使用しないで下さい。**具体的には、nn.Parameter, nn.Module, nn.Sequential, nn.functional以外のnn系のAPIです。
- torchvision等で既に実装されているモデルも使用しないで下さい。

### 提出方法

- 2つのファイルを提出していただきます。
  - テストデータ (x_test) に対する予測ラベルをcsvファイル (ファイル名: submission_pred.csv) で提出してください。
  - それに対応するpythonのコードをsubmission_code.pyとして提出してください (%%writefileコマンドなどを利用してください)。

### 評価方法

- 予測ラベルのt_testに対する精度 (Accuracy) で評価します。
- 定時に採点しLeader Boardを更新します。(採点スケジュールは別アナウンス）
- 締切後の点数を最終的な評価とします。

### データの読み込み

- この部分は修正しないでください
"""

from google.colab import  drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import inspect

nn_except = ["Module", "Parameter", "Sequential", "functional"]
for m in inspect.getmembers(nn):
    if not m[0] in nn_except and m[0][0:2] != "__":
        delattr(nn, m[0]) 


#学習データ
x_train = np.load('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture04_20210506/data/x_train.npy')
t_train = np.load('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture04_20210506/data/t_train.npy')
    
#テストデータ
x_test = np.load('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture04_20210506/data/x_test.npy')



class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = x_train.reshape(-1, 1, 28, 28).astype('float32') / 255
        self.t_train = t_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float), torch.tensor(t_train[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 1, 28, 28).astype('float32') / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)

trainval_data = train_dataset(x_train, t_train)
test_data = test_dataset(x_test)


batch_size = 32

val_size = 10000
train_size = len(trainval_data) - val_size

train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv(nn.Module):
    def __init__(self, filter_shape, function=lambda x: x, stride=(1, 1), padding=0):
      super().__init__()
      fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
      fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]

      self.W = nn.Parameter(
          torch.tensor(
              rng.uniform(
                  low=-np.sqrt(6/fan_in),
                  high=np.sqrt(6/fan_in),
                  size=filter_shape
              ).astype('float32')
          )
      )

      self.b = nn.Parameter(
          torch.tensor(
              np.zeros((filter_shape[0]), dtype='float32')
          )
      )

      self.function = function
      self.stride = stride
      self.padding = padding

    def forward(self, x):
      u = F.conv2d(x, self.W, self.b, self.stride, self.padding)
      return self.function(u)


class Pooling(nn.Module):
    def __init__(self, ksize=(2, 2), stride=(2, 2), padding=0):
      super().__init__()
      self.ksize = ksize
      self.stride = stride
      self.padding = padding

    def forward(self, x):
      return F.avg_pool2d(x, self.ksize, self.stride, self.padding)


class Flatten(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return x.view(x.size()[0], -1)


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, function=lambda x: x):
      super().__init__()
      self.W = nn.Parameter(
          torch.tensor(
              rng.uniform(
                  low = -np.sqrt(6/in_dim),
                  high = np.sqrt(6/out_dim),
                  size = (in_dim, out_dim)
              ).astype('float32')
          )
      )
      
      self.b = nn.Parameter(
          torch.tensor(
              np.zeros(([out_dim]), dtype='float32')
          )
      )

      self.function = function

    def forward(self, x):
      return self.function(torch.matmul(x, self.W) + self.b)

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
      super(MLP, self).__init__()
      self.linear1 = Dense(in_dim, hid_dim)
      self.linear2 = Dense(hid_dim, out_dim)

    def forward(self, x):
      x = F.relu(self.linear1(x))
      x = F.relu(self.linear2(x))
      return x

conv_net = nn.Sequential(
    Conv((32, 1, 3, 3), F.relu, stride=(1, 1), padding=1),    # 28x28x1   -> 28x28x32
    Conv((32, 32, 3, 3), F.relu, stride=(1, 1), padding=1),   # 28x28x32  -> 28x28x32
    Conv((32, 32, 3, 3), F.relu, stride=(1, 1), padding=1),   # 28x28x32  -> 28x28x32
    Pooling((2, 2)),                                          # 28x28x32  -> 14x14x32
    Conv((128, 32, 3, 3), F.relu, stride=(1, 1), padding=1),  # 14x14x32  -> 14x14x128
    Conv((128, 128, 3, 3), F.relu, stride=(1, 1), padding=1), # 14x14x128 -> 14x14x128
    Pooling((2, 2)),                                          # 14x14x128 -> 7x7x128
    Conv((256, 128, 3, 3), F.relu, stride=(1, 1), padding=1), # 14x14x128 -> 14x14x256
    Flatten(),
    # MLP(7*7*256, 1024, 10),
    Dense(7*7*256, 10)
)

n_epochs = 10
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


conv_net.to(device)
optimizer = optim.Adam(conv_net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.85 ** epoch)

for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    train_num = 0
    train_true_num = 0
    valid_num = 0
    valid_true_num = 0

    conv_net.train()
    for x, t in dataloader_train:
        conv_net.zero_grad() 
        x = x.to(device)

        t_hot = torch.eye(10)[t]
        t_hot = t_hot.to(device)
        y = conv_net.forward(x)

        loss = -(t_hot*torch.log_softmax(y, dim=-1)).sum(axis=1).mean() 

        loss.backward()

        optimizer.step()

        pred = y.argmax(1)

        losses_train.append(loss.tolist())

        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        train_num += acc.size()[0]
        train_true_num += acc.sum().item()

    conv_net.eval()
    for x, t in dataloader_valid:
        x = x.to(device)

        t_hot = torch.eye(10)[t]
        t_hot = t_hot.to(device)

        y = conv_net.forward(x)

        loss = -(t_hot*torch.log_softmax(y, dim=-1)).sum(axis=1).mean()

        pred = y.argmax(1)
        
        losses_valid.append(loss.tolist())

        acc = torch.where(t - pred.to("cpu") == 0, torch.ones_like(t), torch.zeros_like(t))
        valid_num += acc.size()[0]
        valid_true_num += acc.sum().item()

    scheduler.step()
    print('EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]'.format(
        epoch+1,
        np.mean(losses_train),
        train_true_num/train_num,
        np.mean(losses_valid),
        valid_true_num/valid_num
    ))

conv_net.eval()

t_pred = []
for x in dataloader_test:

    x = x.to(device)

    y = conv_net.forward(x)

    pred = y.argmax(1).tolist()

    t_pred.extend(pred)

submission = pd.Series(t_pred, name='label')
# submission.to_csv('drive/My Drive/Colab Notebooks/DLBasics2021_colab/Lecture04_20210506/submission_pred.csv', header=True, index_label='id')

