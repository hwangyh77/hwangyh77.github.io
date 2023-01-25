---
layout: post
title:  "Pix2Pix 코드 실습과 설명!!!"
---

```python
from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
!git clone https://github.com/mrzhu-cool/pix2pix-pytorch # 깃 클론
```

    Cloning into 'pix2pix-pytorch'...
    remote: Enumerating objects: 68, done.[K
    remote: Counting objects: 100% (23/23), done.[K
    remote: Compressing objects: 100% (9/9), done.[K
    remote: Total 68 (delta 17), reused 14 (delta 14), pack-reused 45[K
    Unpacking objects: 100% (68/68), 84.94 MiB | 12.21 MiB/s, done.
    


```python
!mkdir 'data' # 폴더 생성
```


```python
!unzip /content/pix2pix-pytorch/dataset/facades.zip  -d /content/data; # 압축 풀기
```
    

# < PIx2PIx >  

pix2pix에 대해 간략하게 말하자면, 왼쪽사진이 입력으로 들어가는 조건 이미지 인데, 이러한 이미지가 입력으로 들어왔을 때 오른쪽과 같이 사진으로 찍은거 같은 이미지가 출력될 수 있도록 하는 것.

이미지는 위에서 본것처럼 입력과 출력형태의 이미지가 두 개가 붙어있는 형태이지만, 학습을 진행할때는 따로따로 봐야되므로 이미지를 네트워크에 넣기 위해 custom dataset을 정의해준다.  

> <Dataset & Dataloader>
> 
> Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있는 객체(iterable)로 감쌉니다.
>
> - Dataset : 전체 dataset을 구성하는 단계로 dataloader를 통해 data를 받아오는 역할을 합니다. 
>
>> <Dataset class에서 반드시 정의해야 하는 method 3가지>
>>* init(self) : 필요한 변수 선언
>>* get_item(self,index) : 만든 리스트의 index에 해당하는 샘플을 데이터셋에서 불러오고 전처리하여 tensor(배열, 행렬) 자료형으로 바꿔서 리턴
>>* len(self) : 학습 데이터 개수 리턴  
>
> - Dataloader : dataset으로부터 dataloader를 생성한다. Dataloader는 Dataset을 batch 기반의 딥러닝모델 학습을 위해 미니배치 형태로 만들어 주는 기능을 하는데, Dataloader를 통해 Dataset의 전체 데이터가 batch size로 나누어져 공급됩니다.  


코드를 보면 이미지는 기본적으로 색상 이미지 형태로 처리할 수 있도록 하였고, 전장에서 봤던 사진을 a를 입력형태의 이미지, b를 출력 형태의 이미지로 지정해놓고 사용할 수 있도록 하였다.


```python
# Custom dataset 생성

class FacadeDataset(Dataset):
    def __init__(self, path2img, direction='b2a', transform=False):
        super().__init__()
        self.direction = direction
        self.path2a = join(path2img, 'a')
        self.path2b = join(path2img, 'b')
        self.img_filenames = [x for x in listdir(self.path2a)]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2a, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2b, self.img_filenames[index])).convert('RGB')
        
        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        if self.direction == 'b2a':
            return b,a
        else:
            return a,b

    def __len__(self):
        return len(self.img_filenames)
     
```

# < Transform >  

pix2pix 모델의 경우 256 x 256 이미지 사이즈를 사용하기 때문에 resize 해줘야 한다. 이 때 normalize를 해주는데 범위를 조정함으로써 step해 나가는 landscape를 안정화시켜서 local optima 문제를 예방하고, 속도 측면에서도 좋아지기 때문이다.  

> - local optima : 중간중간 움푹파인부분이 최소라고 생각되게 만들어지는 문제


```python
# transforms 정의
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((256,256))
])
     
```

### Normalize

resize한 이미지들의 mean과 std로 normalize를 진행해야 한다.(resize한 이미지를 학습할 것이기 때문에 resize한 데이터의  평균(mean)과, 표준편차(standard deviation)를 이용해야 한다.) but 그냥 mean과 std를 0.5로 normalization을 진행하기도 한다.


```python
# 데이터셋 불러오기
path2img = '/content/data/facades/train'
train_ds = FacadeDataset(path2img, transform=transform)
```


```python

```


```python
# 샘플 이미지 확인하기
a,b = train_ds[0]
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(to_pil_image(0.5*a+0.5))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(to_pil_image(0.5*b+0.5))
plt.axis('off')
```




    (-0.5, 255.5, 255.5, -0.5)




    
![image](https://user-images.githubusercontent.com/93988405/213350408-29082f9c-8d4d-4667-9235-ad908b177d68.png)

    


## Dataloader  

이제 데이터셋을 불러와 앞에서 만든 custom dataset에 넣은 것을 train_ds에 저장하고 train_ds를 사용해 dataloder를 만들어 train_dl에 저장한다.


```python
# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
```

# < 모델 구축하기 >

![image](https://user-images.githubusercontent.com/93988405/213350051-f48f36a3-d6b8-4dbc-9b41-033dd28d4a25.png)


pix2pix는 U-net구조를 사용한다. U-net구조란 skip connection을 이용해 왼쪽 사진의 아래쪽 화살표와 같이 기본적으로 인코더 파트에서 나온 출력값을 그대로 가져와 디코더 파트에서 사용한다. 즉 인코더 파트에서 처리하는 low-level information이 디코딩 되는 출력 결과에서도 충분히 활용된다.


```python
# U-Net Down
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

# check
x = torch.randn(16, 3, 256,256, device=device)
model = UNetDown(3,64).to(device)
down_out = model(x)
print(down_out.shape)
```

    torch.Size([16, 64, 128, 128])
    

U-net 아키텍처의 downsampling 모듈입니다. 입력값이 들어오면 convolution layer를 거칠수 있도록 하고 kernel_size = 4, stride = 2, padding = 1 로 설정한다 -> 너비와 높이가 2배씩 감소한다.(Channel size는 증가하도록 만든다)
이후 normalize를 사용할 수 있도록 하고 leakyRelu와 dropout을 사용할 수 있도록 하고 이런 전체 layer를 하나로 묶어 모델로써 사용하는데
<font color='red'>코드에서 init부분의 전체 layer들을 하나로 묶어 forward()함수에서 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 구조이다.</font>


```python
# U-Net Up
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

# check
x = torch.randn(16, 128, 64, 64, device=device)
model = UNetUp(128,64).to(device)
out = model(x,down_out)
print(out.shape)
     
```

    torch.Size([16, 128, 128, 128])
    

upmodeling 코드를 보면 skip connection을 사용하기 때문에 앞의 인코더 파트에서 사용되는 모델의 출력값을 같이 입력으로 받아서 **“def forward에서”**  skip으로 사용할 수 있게 한다. 기본적으로 입력 x가 들어왔을 때 transposed convolution layer를 사용해 너비와 높이를 2배씩 증가시킨다.(Channel size는 감소하도록 만든다) 그 외의 init의 layer 구조는 down과 동일
def forward를 보면 init의 layer들을 하나의 모델로 묶어 입력으로 들어온 x에 대해서 이러한 모델을 거치도록 만들어 주고 그 출력에 특정한 skip를 더해주어서 그것을 최종 output으로 사용한다.
<font color='red'>이때 skip을 더해주는 과정을 channel level에서 합쳐 채널을 두껍게 만든다.</font>


```python
# generator: 가짜 이미지를 생성합니다.
class GeneratorUNet(nn.Module):
  #RGB color이므로 channel = 3
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        #down sampling 이므로 채널크기는 커지고 너비 높이는 절반씩
        self.down1 = UNetDown(in_channels, 64, normalize=False) # 출력 : [64 x 128 x 128]
        self.down2 = UNetDown(64,128)                 # 출력 : [128 x 64 x 64]
        self.down3 = UNetDown(128,256)                # 출력 : [256 x 32 x 32]
        self.down4 = UNetDown(256,512,dropout=0.5)    # 출력 : [512 x 16 x 16]
        self.down5 = UNetDown(512,512,dropout=0.5)    # 출력 : [512 x 8 x 8]
        self.down6 = UNetDown(512,512,dropout=0.5)    # 출력 : [512 x 4 x 4]
        self.down7 = UNetDown(512,512,dropout=0.5)    # 출력 : [512 x 2 x 2]
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5) # 출력 : [512 x 1 x 1]

        # skip-connection을 사용하므로 출력 채널의 크기 x 2 == 다음 입력채널의 크기
        self.up1 = UNetUp(512,512,dropout=0.5)    # 출력 : [1024 x 2 x 2]
        self.up2 = UNetUp(1024,512,dropout=0.5)   # 출력 : [1024 x 4 x 4]
        self.up3 = UNetUp(1024,512,dropout=0.5)   # 출력 : [1024 x 8 x 8]
        self.up4 = UNetUp(1024,512,dropout=0.5)   # 출력 : [1024 x 16 x 16]
        self.up5 = UNetUp(1024,256)               # 출력 : [512 x 32 x 32]
        self.up6 = UNetUp(512,128)                # 출력 : [256 x 64 x 64]
        self.up7 = UNetUp(256,64)                 # 출력 : [128 x 128 x 128]
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),  # 출력 : [3 x 256 x 256]
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7) # 보틀넥 정중앙의 젤 작은 정사각형

        u1 = self.up1(d8,d7) # ,오른쪽값은 skip값으로 그대로 더해지는... 즉 upsampling block에 d8을 넣어 처리된 결과에 d7을 그대로 더한다.
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7) # 결국 입력이미지와 동일한 차원의 출력이미지가 된다.

        return u8

# check
x = torch.randn(16,3,256,256,device=device)
model = GeneratorUNet().to(device)
out = model(x)
print(out.shape)
```

    torch.Size([16, 3, 256, 256])
    

이제 U-net down, up 모듈을 사용해 Generator 아키텍처를 만들었다.
처음 input channel은 RGB color이므로 3이되고 down sampling을 진행하여 channel size는 키워준다. 주석에 써놓았듯이 채널의 크기는 2배씩 증가하고 너비와 높이는 2배씩 감소한다. 그래서 최종적으로 512 x 1 x 1형태가 되고 이제 디코더 부분에서 up sampling을 진행한다. skip connection을 사용하므로 다음 입력 채널의 크기는 (출력채널의 크기 x 2) 가 된다. 그렇게 쭉 layer를 거치면 3 x 256 x 256 형태 즉 입력이미지와 동일한 차원의 출력이미지가 만들어진다. 오른쪽의 forward 연산 부분의 up 부분의 처음 부분을 보면 U-net 구조의 사진처럼 upsampling block에 d8을 넣어 처리된 결과에 d7을 그대로 더하는 방식을 사용해 skip-connection을 구현하였습니다.

![image](https://user-images.githubusercontent.com/93988405/213350104-bafd9d35-e32d-43ab-b1d7-b19003007813.png)


Pix2Pix는 conditional GAN을 사용하기 때문에 사진과 같이 입력으로 들어오는 조건이미지(x)와 그에따른 y 도메인 이미지(G(x))가 같이 들어와야된다. 따라서 항상 discriminator가 입력으로 받는 것은 조건+가짜이미지 혹은 조건 + 진짜 이미지이다. 




```python
class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

# check
x = torch.randn(16,64,128,128,device=device)
model = Dis_block(64,128).to(device)
out = model(x)
print(out.shape)
```

    torch.Size([16, 128, 64, 64])
    


```python
# Discriminator은 patch gan을 사용합니다.
# Patch Gan: 이미지를 16x16의 패치로 분할하여 각 패치가 진짜인지 가짜인지 식별합니다.
# high-frequency에서 정확도가 향상됩니다.

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False) # 출력 : [64 x 128 x 128]
        self.stage_2 = Dis_block(64,128)    # 출력 : [128 x 64 x 64]
        self.stage_3 = Dis_block(128,256)   # 출력 : [256 x 32 x 32]
        self.stage_4 = Dis_block(256,512)   # 출력 : [512 x 16 x 16]

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x
# check
x = torch.randn(16,3,256,256,device=device)
model = Discriminator().to(device)
out = model(x,x)
print(out.shape)
```

    torch.Size([16, 1, 16, 16])
    

Discriminator는
PatchGAN을 사용하므로 원하는 patch 사이즈인 16x16으로 만들기 위해 4개의 stage를 지나며 convolution 연산을 수행하여 16x16형태의 패치를 생성한다.
코드의 stage1을 보면 입력 이미지가 2개이니깐 channel*2 형태이고, forward 함수를 보면 입력받는 이미지가 2개입니다. (입력 이미지가 2개라는 것은 2개의 이미지가 한쌍을 이루어 입력으로 들어온다는 것)
<font color='red'> stage를 거칠수록 채널수는 늘어나고 이미지의 높이와 너비는 반씩 줄어들게되어 stage4에서는 512 x 16 x 16형태가 되고 이를 self.patch를 통해 1 x 16 x 16으로(압축)만들어주게 되고 마지막으로 sigmoid를 취해 각 픽셀값을 sigmoid형태로 해줍니다.</font>


```python
model_gen = GeneratorUNet().to(device)
model_dis = Discriminator().to(device)
```


```python
# 가중치 초기화
def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)


# 가중치 초기화 적용
model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);
```

# < 학습 >

- **BCEloss** : binary cross entropy loss(BCELoss)로 이진 분류에 특화됐다.
BCELoss에서는 CrossEntropyLoss와 같이 softmax를 포함한 것이 아닌, Cross Entropy만 구합니다. 따라서, 이 loss를 사용하는 경우엔 softmax 또는 다른 activation function을 따로 적용해주어야 합니다.

- **MSEloss** : Mean Squared Error loss로  image 간의 차이나 segmentation 에서는 mask 간의 차이를 구하기 위해 많이 사용


```python
# 손실함수
#loss_func_gan = nn.BCELoss()
loss_func_gan = nn.MSELoss()
loss_func_pix = nn.L1Loss()

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1,256//2**4,256//2**4)

# 최적화 파라미터
from torch import optim
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))
```

실습을 할 때 loss 함수로는 MSEloss를 사용하였고 나중에 generator loss에서 L1 loss에 곱할 가중치와 에포크를 100으로 설정하여 학습을 진행하였다. 


```python
# 학습
model_gen.train()
model_dis.train()

batch_count = 0
num_epochs = 100
start_time = time.time()

loss_hist = {'gen':[],
             'dis':[]}

for epoch in range(num_epochs):
    for a, b in train_dl:
        ba_si = a.size(0)

        # real image
        real_a = a.to(device) # 입력이미지(그린이미지)
        real_b = b.to(device) # 출력이미지(사진)

        # patch label
        real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

        ### generator 학습 ###
        model_gen.zero_grad()

        fake_b = model_gen(real_a) # 가짜 이미지 생성
        out_dis = model_dis(fake_b, real_b) # 가짜 이미지 식별

        gen_loss = loss_func_gan(out_dis, real_label)
        pixel_loss = loss_func_pix(fake_b, real_b)  # 만들어진 결과가 실제 사진과 유사할수 있도록(L1 loss로 학습)

        g_loss = gen_loss + lambda_pixel * pixel_loss
        g_loss.backward() # backpropagation -> loss를 역전파한다(손실의 변화도 저장)
        opt_gen.step()    # .step()을 호출해 역전파 단계에서 수집된 변화도로 매개변수 조정

        ### discriminator 학습 ###
        model_dis.zero_grad()

        out_dis = model_dis(real_b, real_a) # 진짜 이미지 식별
        real_loss = loss_func_gan(out_dis,real_label)
        
        out_dis = model_dis(fake_b.detach(), real_a) # 가짜 이미지 식별
        fake_loss = loss_func_gan(out_dis,fake_label)

        d_loss = (real_loss + fake_loss) / 2.
        
        # 판별자(discriminator 업데이트)
        d_loss.backward()
        opt_dis.step()

        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

        batch_count += 1
        if batch_count % 100 == 0:
            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
```

    Epoch: 7, G_Loss: 32.751335, D_Loss: 0.012031, time: 1.50 min
    Epoch: 15, G_Loss: 31.956936, D_Loss: 0.004464, time: 3.04 min
    Epoch: 23, G_Loss: 28.797182, D_Loss: 0.004848, time: 4.61 min
    Epoch: 30, G_Loss: 28.570894, D_Loss: 0.000893, time: 6.17 min
    Epoch: 38, G_Loss: 25.091671, D_Loss: 0.007483, time: 7.72 min
    Epoch: 46, G_Loss: 22.723743, D_Loss: 0.001539, time: 9.26 min
    Epoch: 53, G_Loss: 21.941488, D_Loss: 0.000706, time: 10.83 min
    Epoch: 61, G_Loss: 21.969990, D_Loss: 0.000505, time: 12.37 min
    Epoch: 69, G_Loss: 21.019608, D_Loss: 0.000349, time: 13.91 min
    Epoch: 76, G_Loss: 19.570671, D_Loss: 0.000689, time: 15.46 min
    Epoch: 84, G_Loss: 20.565851, D_Loss: 0.000278, time: 17.02 min
    Epoch: 92, G_Loss: 18.415768, D_Loss: 0.000457, time: 18.56 min
    Epoch: 99, G_Loss: 20.046328, D_Loss: 0.205631, time: 20.10 min
    

discriminator loss 결과값의 마지막을 보면 discriminator의 loss값이 항상 작다가 99번째쯤에 500배쯤 커진 것을 볼 수 있었는데 이것을 보아 에포크를 늘리면 좀더 좋은 학습 결과가 나올것이라 예상할 수 있다.

학습을 진행하기 위해 real_a는 입력 이미지, real_b는 출력 형태의 사진 이미지로 설정하였고 conditionalGAN의 특성상 generator와 discriminator 둘 다 label을 사용해 학습하므로 real_label과 fake_label을 만들어 주었다.

신경망을 최적화하기 위해서 논문에서 나온대로 D와 G를 번갈아가며 gradient descent step을 진행하였다.

### Generator 학습 부분  

입력 이미지를 generator model에 넣어 가짜 이미지를 생성한 것을 fake_b에 저장하고, 생성된 이미지와 출력 형태의 사진을 discriminator model에 넣은, 즉 가짜 이미지를 식별한걸 out_dis에 저장합니다. out_dis와 real_label은 MSEloss로, 만들어진 결과가 실제 사진과 유사할 수 있도록 fake_b와 real_b는 L1 loss를 통해 학습합니다. 그리하여 최종 loss는 generator loss는 gen_loss + pixel_loss * 가중치의 형태로 구성.

### Discriminator 학습 부분  

real_b와 real_a로 진짜 이미지를 식별한 것을 out_dis에 저장하여 real_label과 MSEloss로 학습한 것을 real_loss에 저장하고, fake_b와 real_a로 가짜 이미지를 식별한걸 out_dis에 저장하여 fake_label과 MSEloss로 학습한 것을 fake_loss에 저장했다. 따라서 discciminator loss는 real_loss 와 fake_loss를 더하는 형식으로 구성하였는데, 이는 discriminator가 가짜 이미지를 판별해 내려면 진짜 이미지에 대한 판별 데이터도 있어야 하기 때문이다. <font color='green'>(discriminator의 loss식을 직관적으로 보자면 진짜 사진에 대해 진짜인지 가짜인지와 생성된 사진에 대해 진짜인지 가짜인지의 loss합이라 보면 될것같다.)</font> 또한 D를 학습시키는 동안 discriminator loss를 절반으로 나눠서, G의 학습 속도에 맞게 하였다.


```python
# loss history
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![image](https://user-images.githubusercontent.com/93988405/213350535-63319da9-5d5c-42a4-a6f3-25293b762024.png)

    



```python
# 가중치 저장
path2models = './models/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)
```


```python
## Generator가 생성한 가짜 이미지 확인하기
```


```python
# 가중치 불러오기
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)
```




    <All keys matched successfully>




```python
# evaluation model
model_gen.eval()

# 가짜 이미지 생성
with torch.no_grad():
    for a,b in train_dl:
        fake_imgs = model_gen(a.to(device)).detach().cpu()
        real_imgs = b
        break
```


```python
# 가짜 이미지 시각화
plt.figure(figsize=(10,10))

for ii in range(0,16,2):
    plt.subplot(4,4,ii+1)
    plt.imshow(to_pil_image(0.5*real_imgs[ii]+0.5))
    plt.axis('off')
    plt.subplot(4,4,ii+2)
    plt.imshow(to_pil_image(0.5*fake_imgs[ii]+0.5))
    plt.axis('off')
```


    
![image](https://user-images.githubusercontent.com/93988405/213350570-09ec13aa-86f9-4887-94bd-ea6b15c2a841.png)

    

