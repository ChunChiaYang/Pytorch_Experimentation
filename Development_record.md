CIFAR-10_Classification
===
作者: 楊峻嘉

實作過程
---
### 一、LeNet架構實作
1. 定義網路:
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
2. 正向傳播:
x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
x=F.max_pool2d(F.relu(self.conv2(x)), 2)
x=x.view(x.size()[0],-1)
x=F.relu(self.fc1(x))
x=F.relu(self.fc2(x))
x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:2.179
[1, 4000] loss:1.893
[1, 6000] loss:1.744
[1, 8000] loss:1.596
[1, 10000] loss:1.557
[1, 12000] loss:1.478
[2, 2000] loss:1.413
[2, 4000] loss:1.371
[2, 6000] loss:1.356
[2, 8000] loss:1.348
[2, 10000] loss:1.321
[2, 12000] loss:1.311
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 55 %

### 二、加入一層BatchNorm2d
1. 定義網路:
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (norm1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
2. 正向傳播:
x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
x=self.norm1(x) 
x=F.max_pool2d(F.relu(self.conv2(x)), 2)
x=x.view(x.size()[0],-1)
x=F.relu(self.fc1(x))
x=F.relu(self.fc2(x))
x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:2.084
[1, 4000] loss:1.727
[1, 6000] loss:1.602
[1, 8000] loss:1.506
[1, 10000] loss:1.459
[1, 12000] loss:1.413
[2, 2000] loss:1.336
[2, 4000] loss:1.306
[2, 6000] loss:1.299
[2, 8000] loss:1.280
[2, 10000] loss:1.263
[2, 12000] loss:1.203
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 56 %

### 三、加入兩層BatchNorm2d
1. 定義網路:
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (norm1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (norm2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
2. 正向傳播:
x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
x=self.norm1(x)
x=F.max_pool2d(F.relu(self.conv2(x)), 2)
x=self.norm2(x)
x=x.view(x.size()[0],-1)
x=F.relu(self.fc1(x))
x=F.relu(self.fc2(x))
x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:2.011
[1, 4000] loss:1.691
[1, 6000] loss:1.581
[1, 8000] loss:1.512
[1, 10000] loss:1.443
[1, 12000] loss:1.375
[2, 2000] loss:1.299
[2, 4000] loss:1.293
[2, 6000] loss:1.299
[2, 8000] loss:1.257
[2, 10000] loss:1.228
[2, 12000] loss:1.230
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 56 %

### 四、改變kernal_size(3,3)
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:1.806
[1, 4000] loss:1.537
[1, 6000] loss:1.392
[1, 8000] loss:1.323
[1, 10000] loss:1.239
[1, 12000] loss:1.187
[2, 2000] loss:1.083
[2, 4000] loss:1.024
[2, 6000] loss:1.021
[2, 8000] loss:0.998
[2, 10000] loss:0.987
[2, 12000] loss:0.949
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 66 %

### 五、將epoch改為3
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:0.862
[1, 4000] loss:0.848
[1, 6000] loss:0.843
[1, 8000] loss:0.817
[1, 10000] loss:0.822
[1, 12000] loss:0.838
[2, 2000] loss:0.686
[2, 4000] loss:0.697
[2, 6000] loss:0.705
[2, 8000] loss:0.716
[2, 10000] loss:0.723
[2, 12000] loss:0.718
[3, 2000] loss:0.582
[3, 4000] loss:0.593
[3, 6000] loss:0.614
[3, 8000] loss:0.623
[3, 10000] loss:0.604
[3, 12000] loss:0.612
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 71 %

### 六、將SGD的momentum參數改為0.5
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:0.412
[1, 4000] loss:0.374
[1, 6000] loss:0.364
[1, 8000] loss:0.354
[1, 10000] loss:0.346
[1, 12000] loss:0.348
[2, 2000] loss:0.275
[2, 4000] loss:0.279
[2, 6000] loss:0.280
[2, 8000] loss:0.294
[2, 10000] loss:0.291
[2, 12000] loss:0.295
[3, 2000] loss:0.231
[3, 4000] loss:0.225
[3, 6000] loss:0.254
[3, 8000] loss:0.257
[3, 10000] loss:0.226
[3, 12000] loss:0.246
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.5)
5. 準確率: 74 %

### 七、加入兩層Dropout
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.dropout(x)
        x=F.relu(self.fc2(x))
        x=F.dropout(x)
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:2.095
[1, 4000] loss:1.781
[1, 6000] loss:1.655
[1, 8000] loss:1.567
[1, 10000] loss:1.507
[1, 12000] loss:1.438
[2, 2000] loss:1.394
[2, 4000] loss:1.336
[2, 6000] loss:1.311
[2, 8000] loss:1.309
[2, 10000] loss:1.293
[2, 12000] loss:1.267
[3, 2000] loss:1.071
[3, 4000] loss:1.028
[3, 6000] loss:1.020
[3, 8000] loss:0.999
[3, 10000] loss:0.966
[3, 12000] loss:0.960
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.5)
5. 準確率: 65 %

### 八、把SGD的momentum參數改為0.9
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.dropout(x)
        x=F.relu(self.fc2(x))
        x=F.dropout(x)
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:1.193
[1, 4000] loss:1.122
[1, 6000] loss:1.069
[1, 8000] loss:1.051
[1, 10000] loss:0.983
[1, 12000] loss:0.950
[2, 2000] loss:0.849
[2, 4000] loss:0.842
[2, 6000] loss:0.851
[2, 8000] loss:0.816
[2, 10000] loss:0.829
[2, 12000] loss:0.799
[3, 2000] loss:0.686
[3, 4000] loss:0.686
[3, 6000] loss:0.671
[3, 8000] loss:0.700
[3, 10000] loss:0.713
[3, 12000] loss:0.711
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 71 %

### 九、將epoch改為4
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.dropout(x)
        x=F.relu(self.fc2(x))
        x=F.dropout(x)
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:1.111
[1, 4000] loss:1.090
[1, 6000] loss:1.105
[1, 8000] loss:1.057
[1, 10000] loss:1.046
[1, 12000] loss:1.019
[2, 2000] loss:0.980
[2, 4000] loss:0.988
[2, 6000] loss:0.961
[2, 8000] loss:0.922
[2, 10000] loss:0.954
[2, 12000] loss:0.909
[3, 2000] loss:0.858
[3, 4000] loss:0.881
[3, 6000] loss:0.876
[3, 8000] loss:0.875
[3, 10000] loss:0.868
[3, 12000] loss:0.881
[4, 2000] loss:0.797
[4, 4000] loss:0.824
[4, 6000] loss:0.804
[4, 8000] loss:0.788
[4, 10000] loss:0.825
[4, 12000] loss:0.814
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 68 %

### 十、將Dropout_rate改為0.3
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,p=0.3)
        x=F.relu(self.fc2(x))
        x=F.dropout(x,p=0.3)
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:1.838
[1, 4000] loss:1.530
[1, 6000] loss:1.389
[1, 8000] loss:1.292
[1, 10000] loss:1.229
[1, 12000] loss:1.193
[2, 2000] loss:1.040
[2, 4000] loss:1.037
[2, 6000] loss:1.002
[2, 8000] loss:1.003
[2, 10000] loss:0.936
[2, 12000] loss:0.961
[3, 2000] loss:0.824
[3, 4000] loss:0.814
[3, 6000] loss:0.823
[3, 8000] loss:0.829
[3, 10000] loss:0.819
[3, 12000] loss:0.812
[4, 2000] loss:0.676
[4, 4000] loss:0.678
[4, 6000] loss:0.712
[4, 8000] loss:0.705
[4, 10000] loss:0.718
[4, 12000] loss:0.709
4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 72 %

### 十一、將最佳化器改為Adam並把epoch改為5
1. 定義網路:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1600, out_features=1000, bias=True)
  (fc2): Linear(in_features=1000, out_features=500, bias=True)
  (fc3): Linear(in_features=500, out_features=10, bias=True)
)
2. 正向傳播:
x=F.relu(self.conv1(x))
        x=self.norm1(x)
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.norm2(x)
        x=self.pool2(x)
        x=x.view(x.size()[0],-1)
        x=F.dropout(self.fc1(x),p=0.3)
        x=F.relu(x)
        x=F.dropout(self.fc2(x),p=0.3)
        x=F.relu(x)
        x=self.fc3(x)
3. 損失函數:
nn.CrossEntropyLoss()
[1, 2000] loss:1.183
[1, 4000] loss:1.114
[1, 6000] loss:1.094
[1, 8000] loss:1.073
[1, 10000] loss:1.034
[1, 12000] loss:1.025
[2, 2000] loss:0.886
[2, 4000] loss:0.883
[2, 6000] loss:0.902
[2, 8000] loss:0.882
[2, 10000] loss:0.846
[2, 12000] loss:0.866
[3, 2000] loss:0.728
[3, 4000] loss:0.725
[3, 6000] loss:0.733
[3, 8000] loss:0.741
[3, 10000] loss:0.778
[3, 12000] loss:0.752
[4, 2000] loss:0.605
[4, 4000] loss:0.619
[4, 6000] loss:0.626
[4, 8000] loss:0.656
[4, 10000] loss:0.681
[4, 12000] loss:0.646
[5, 2000] loss:0.517
[5, 4000] loss:0.527
[5, 6000] loss:0.555
[5, 8000] loss:0.565
[5, 10000] loss:0.568
[5, 12000] loss:0.585
4. 最佳化器:
optimizer=optim.Adam(net.parameters(),lr=0.001)
5. 準確率: 72 %

