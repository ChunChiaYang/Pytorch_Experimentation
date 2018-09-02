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
![](https://i.imgur.com/ArYYOEt.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 54 %

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
![](https://i.imgur.com/m2PNS3y.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 59 %

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
![](https://i.imgur.com/OaYXmux.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 58 %

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
![](https://i.imgur.com/q0QZCRu.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 69 %

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
![](https://i.imgur.com/uiNmRC4.png)


4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 74 %

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
![](https://i.imgur.com/otQdDkw.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.5)
5. 準確率: 69 %

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
![](https://i.imgur.com/b6hsOCQ.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.5)
5. 準確率: 77 %

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
![](https://i.imgur.com/ZDNESaW.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 74 %

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
![](https://i.imgur.com/JYcl0bc.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 74 %

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
![](https://i.imgur.com/l67WJFi.png)

4. 最佳化器:
optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
5. 準確率: 73 %

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
3. 訓練損失&驗證損失:
nn.CrossEntropyLoss()
![](https://i.imgur.com/dizfxBC.png)
4. 最佳化器:
optimizer=optim.Adam(net.parameters(),lr=0.001)
5. 準確率: 71 %

