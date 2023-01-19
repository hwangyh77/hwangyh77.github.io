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

    Archive:  /content/pix2pix-pytorch/dataset/facades.zip
       creating: /content/data/facades/
       creating: /content/data/facades/test/
       creating: /content/data/facades/test/a/
      inflating: /content/data/facades/test/a/cmp_b0001.png  
      inflating: /content/data/facades/test/a/cmp_b0002.png  
      inflating: /content/data/facades/test/a/cmp_b0003.png  
      inflating: /content/data/facades/test/a/cmp_b0004.png  
      inflating: /content/data/facades/test/a/cmp_b0013.png  
      inflating: /content/data/facades/test/a/cmp_b0023.png  
      inflating: /content/data/facades/test/a/cmp_b0026.png  
      inflating: /content/data/facades/test/a/cmp_b0028.png  
      inflating: /content/data/facades/test/a/cmp_b0034.png  
      inflating: /content/data/facades/test/a/cmp_b0035.png  
      inflating: /content/data/facades/test/a/cmp_b0047.png  
      inflating: /content/data/facades/test/a/cmp_b0049.png  
      inflating: /content/data/facades/test/a/cmp_b0054.png  
      inflating: /content/data/facades/test/a/cmp_b0055.png  
      inflating: /content/data/facades/test/a/cmp_b0057.png  
      inflating: /content/data/facades/test/a/cmp_b0061.png  
      inflating: /content/data/facades/test/a/cmp_b0068.png  
      inflating: /content/data/facades/test/a/cmp_b0071.png  
      inflating: /content/data/facades/test/a/cmp_b0073.png  
      inflating: /content/data/facades/test/a/cmp_b0075.png  
      inflating: /content/data/facades/test/a/cmp_b0077.png  
      inflating: /content/data/facades/test/a/cmp_b0078.png  
      inflating: /content/data/facades/test/a/cmp_b0081.png  
      inflating: /content/data/facades/test/a/cmp_b0084.png  
      inflating: /content/data/facades/test/a/cmp_b0085.png  
      inflating: /content/data/facades/test/a/cmp_b0087.png  
      inflating: /content/data/facades/test/a/cmp_b0089.png  
      inflating: /content/data/facades/test/a/cmp_b0090.png  
      inflating: /content/data/facades/test/a/cmp_b0093.png  
      inflating: /content/data/facades/test/a/cmp_b0095.png  
      inflating: /content/data/facades/test/a/cmp_b0096.png  
      inflating: /content/data/facades/test/a/cmp_b0103.png  
      inflating: /content/data/facades/test/a/cmp_b0105.png  
      inflating: /content/data/facades/test/a/cmp_b0112.png  
      inflating: /content/data/facades/test/a/cmp_b0113.png  
      inflating: /content/data/facades/test/a/cmp_b0121.png  
      inflating: /content/data/facades/test/a/cmp_b0122.png  
      inflating: /content/data/facades/test/a/cmp_b0124.png  
      inflating: /content/data/facades/test/a/cmp_b0125.png  
      inflating: /content/data/facades/test/a/cmp_b0126.png  
      inflating: /content/data/facades/test/a/cmp_b0127.png  
      inflating: /content/data/facades/test/a/cmp_b0135.png  
      inflating: /content/data/facades/test/a/cmp_b0141.png  
      inflating: /content/data/facades/test/a/cmp_b0145.png  
      inflating: /content/data/facades/test/a/cmp_b0146.png  
      inflating: /content/data/facades/test/a/cmp_b0148.png  
      inflating: /content/data/facades/test/a/cmp_b0149.png  
      inflating: /content/data/facades/test/a/cmp_b0150.png  
      inflating: /content/data/facades/test/a/cmp_b0154.png  
      inflating: /content/data/facades/test/a/cmp_b0156.png  
      inflating: /content/data/facades/test/a/cmp_b0157.png  
      inflating: /content/data/facades/test/a/cmp_b0162.png  
      inflating: /content/data/facades/test/a/cmp_b0164.png  
      inflating: /content/data/facades/test/a/cmp_b0165.png  
      inflating: /content/data/facades/test/a/cmp_b0166.png  
      inflating: /content/data/facades/test/a/cmp_b0167.png  
      inflating: /content/data/facades/test/a/cmp_b0168.png  
      inflating: /content/data/facades/test/a/cmp_b0172.png  
      inflating: /content/data/facades/test/a/cmp_b0174.png  
      inflating: /content/data/facades/test/a/cmp_b0184.png  
      inflating: /content/data/facades/test/a/cmp_b0185.png  
      inflating: /content/data/facades/test/a/cmp_b0187.png  
      inflating: /content/data/facades/test/a/cmp_b0188.png  
      inflating: /content/data/facades/test/a/cmp_b0190.png  
      inflating: /content/data/facades/test/a/cmp_b0192.png  
      inflating: /content/data/facades/test/a/cmp_b0199.png  
      inflating: /content/data/facades/test/a/cmp_b0200.png  
      inflating: /content/data/facades/test/a/cmp_b0202.png  
      inflating: /content/data/facades/test/a/cmp_b0204.png  
      inflating: /content/data/facades/test/a/cmp_b0206.png  
      inflating: /content/data/facades/test/a/cmp_b0207.png  
      inflating: /content/data/facades/test/a/cmp_b0214.png  
      inflating: /content/data/facades/test/a/cmp_b0218.png  
      inflating: /content/data/facades/test/a/cmp_b0219.png  
      inflating: /content/data/facades/test/a/cmp_b0227.png  
      inflating: /content/data/facades/test/a/cmp_b0231.png  
      inflating: /content/data/facades/test/a/cmp_b0233.png  
      inflating: /content/data/facades/test/a/cmp_b0234.png  
      inflating: /content/data/facades/test/a/cmp_b0235.png  
      inflating: /content/data/facades/test/a/cmp_b0239.png  
      inflating: /content/data/facades/test/a/cmp_b0241.png  
      inflating: /content/data/facades/test/a/cmp_b0242.png  
      inflating: /content/data/facades/test/a/cmp_b0246.png  
      inflating: /content/data/facades/test/a/cmp_b0247.png  
      inflating: /content/data/facades/test/a/cmp_b0250.png  
      inflating: /content/data/facades/test/a/cmp_b0258.png  
      inflating: /content/data/facades/test/a/cmp_b0260.png  
      inflating: /content/data/facades/test/a/cmp_b0271.png  
      inflating: /content/data/facades/test/a/cmp_b0272.png  
      inflating: /content/data/facades/test/a/cmp_b0275.png  
      inflating: /content/data/facades/test/a/cmp_b0280.png  
      inflating: /content/data/facades/test/a/cmp_b0283.png  
      inflating: /content/data/facades/test/a/cmp_b0285.png  
      inflating: /content/data/facades/test/a/cmp_b0286.png  
      inflating: /content/data/facades/test/a/cmp_b0288.png  
      inflating: /content/data/facades/test/a/cmp_b0290.png  
      inflating: /content/data/facades/test/a/cmp_b0291.png  
      inflating: /content/data/facades/test/a/cmp_b0292.png  
      inflating: /content/data/facades/test/a/cmp_b0293.png  
      inflating: /content/data/facades/test/a/cmp_b0294.png  
      inflating: /content/data/facades/test/a/cmp_b0298.png  
      inflating: /content/data/facades/test/a/cmp_b0300.png  
      inflating: /content/data/facades/test/a/cmp_b0302.png  
      inflating: /content/data/facades/test/a/cmp_b0305.png  
      inflating: /content/data/facades/test/a/cmp_b0306.png  
      inflating: /content/data/facades/test/a/cmp_b0313.png  
      inflating: /content/data/facades/test/a/cmp_b0315.png  
      inflating: /content/data/facades/test/a/cmp_b0319.png  
      inflating: /content/data/facades/test/a/cmp_b0325.png  
      inflating: /content/data/facades/test/a/cmp_b0331.png  
      inflating: /content/data/facades/test/a/cmp_b0334.png  
      inflating: /content/data/facades/test/a/cmp_b0340.png  
      inflating: /content/data/facades/test/a/cmp_b0341.png  
      inflating: /content/data/facades/test/a/cmp_b0343.png  
      inflating: /content/data/facades/test/a/cmp_b0345.png  
      inflating: /content/data/facades/test/a/cmp_b0349.png  
      inflating: /content/data/facades/test/a/cmp_b0352.png  
      inflating: /content/data/facades/test/a/cmp_b0353.png  
      inflating: /content/data/facades/test/a/cmp_b0355.png  
      inflating: /content/data/facades/test/a/cmp_b0360.png  
      inflating: /content/data/facades/test/a/cmp_b0363.png  
      inflating: /content/data/facades/test/a/cmp_b0364.png  
      inflating: /content/data/facades/test/a/cmp_b0365.png  
      inflating: /content/data/facades/test/a/cmp_b0368.png  
      inflating: /content/data/facades/test/a/cmp_b0370.png  
      inflating: /content/data/facades/test/a/cmp_b0372.png  
      inflating: /content/data/facades/test/a/cmp_b0373.png  
      inflating: /content/data/facades/test/a/cmp_b0376.png  
      inflating: /content/data/facades/test/a/cmp_b0377.png  
      inflating: /content/data/facades/test/a/cmp_x0003.png  
      inflating: /content/data/facades/test/a/cmp_x0006.png  
      inflating: /content/data/facades/test/a/cmp_x0007.png  
      inflating: /content/data/facades/test/a/cmp_x0009.png  
      inflating: /content/data/facades/test/a/cmp_x0011.png  
      inflating: /content/data/facades/test/a/cmp_x0013.png  
      inflating: /content/data/facades/test/a/cmp_x0023.png  
      inflating: /content/data/facades/test/a/cmp_x0024.png  
      inflating: /content/data/facades/test/a/cmp_x0026.png  
      inflating: /content/data/facades/test/a/cmp_x0027.png  
      inflating: /content/data/facades/test/a/cmp_x0032.png  
      inflating: /content/data/facades/test/a/cmp_x0033.png  
      inflating: /content/data/facades/test/a/cmp_x0039.png  
      inflating: /content/data/facades/test/a/cmp_x0040.png  
      inflating: /content/data/facades/test/a/cmp_x0041.png  
      inflating: /content/data/facades/test/a/cmp_x0046.png  
      inflating: /content/data/facades/test/a/cmp_x0047.png  
      inflating: /content/data/facades/test/a/cmp_x0053.png  
      inflating: /content/data/facades/test/a/cmp_x0057.png  
      inflating: /content/data/facades/test/a/cmp_x0059.png  
      inflating: /content/data/facades/test/a/cmp_x0060.png  
      inflating: /content/data/facades/test/a/cmp_x0070.png  
      inflating: /content/data/facades/test/a/cmp_x0073.png  
      inflating: /content/data/facades/test/a/cmp_x0076.png  
      inflating: /content/data/facades/test/a/cmp_x0077.png  
      inflating: /content/data/facades/test/a/cmp_x0080.png  
      inflating: /content/data/facades/test/a/cmp_x0083.png  
      inflating: /content/data/facades/test/a/cmp_x0084.png  
      inflating: /content/data/facades/test/a/cmp_x0086.png  
      inflating: /content/data/facades/test/a/cmp_x0093.png  
      inflating: /content/data/facades/test/a/cmp_x0096.png  
      inflating: /content/data/facades/test/a/cmp_x0097.png  
      inflating: /content/data/facades/test/a/cmp_x0100.png  
      inflating: /content/data/facades/test/a/cmp_x0101.png  
      inflating: /content/data/facades/test/a/cmp_x0105.png  
      inflating: /content/data/facades/test/a/cmp_x0107.png  
      inflating: /content/data/facades/test/a/cmp_x0114.png  
      inflating: /content/data/facades/test/a/cmp_x0116.png  
      inflating: /content/data/facades/test/a/cmp_x0117.png  
      inflating: /content/data/facades/test/a/cmp_x0119.png  
      inflating: /content/data/facades/test/a/cmp_x0126.png  
      inflating: /content/data/facades/test/a/cmp_x0127.png  
      inflating: /content/data/facades/test/a/cmp_x0128.png  
      inflating: /content/data/facades/test/a/cmp_x0129.png  
      inflating: /content/data/facades/test/a/cmp_x0130.png  
      inflating: /content/data/facades/test/a/cmp_x0133.png  
      inflating: /content/data/facades/test/a/cmp_x0142.png  
      inflating: /content/data/facades/test/a/cmp_x0143.png  
      inflating: /content/data/facades/test/a/cmp_x0147.png  
      inflating: /content/data/facades/test/a/cmp_x0151.png  
      inflating: /content/data/facades/test/a/cmp_x0156.png  
      inflating: /content/data/facades/test/a/cmp_x0157.png  
      inflating: /content/data/facades/test/a/cmp_x0159.png  
      inflating: /content/data/facades/test/a/cmp_x0160.png  
      inflating: /content/data/facades/test/a/cmp_x0161.png  
      inflating: /content/data/facades/test/a/cmp_x0163.png  
      inflating: /content/data/facades/test/a/cmp_x0164.png  
      inflating: /content/data/facades/test/a/cmp_x0169.png  
      inflating: /content/data/facades/test/a/cmp_x0177.png  
      inflating: /content/data/facades/test/a/cmp_x0183.png  
      inflating: /content/data/facades/test/a/cmp_x0184.png  
      inflating: /content/data/facades/test/a/cmp_x0186.png  
      inflating: /content/data/facades/test/a/cmp_x0188.png  
      inflating: /content/data/facades/test/a/cmp_x0197.png  
      inflating: /content/data/facades/test/a/cmp_x0198.png  
      inflating: /content/data/facades/test/a/cmp_x0205.png  
      inflating: /content/data/facades/test/a/cmp_x0207.png  
      inflating: /content/data/facades/test/a/cmp_x0210.png  
      inflating: /content/data/facades/test/a/cmp_x0213.png  
      inflating: /content/data/facades/test/a/cmp_x0214.png  
      inflating: /content/data/facades/test/a/cmp_x0216.png  
      inflating: /content/data/facades/test/a/cmp_x0217.png  
      inflating: /content/data/facades/test/a/cmp_x0220.png  
      inflating: /content/data/facades/test/a/cmp_x0221.png  
      inflating: /content/data/facades/test/a/cmp_x0223.png  
      inflating: /content/data/facades/test/a/cmp_x0224.png  
      inflating: /content/data/facades/test/a/cmp_x0226.png  
       creating: /content/data/facades/test/b/
      inflating: /content/data/facades/test/b/cmp_b0001.png  
      inflating: /content/data/facades/test/b/cmp_b0002.png  
      inflating: /content/data/facades/test/b/cmp_b0003.png  
      inflating: /content/data/facades/test/b/cmp_b0004.png  
      inflating: /content/data/facades/test/b/cmp_b0013.png  
      inflating: /content/data/facades/test/b/cmp_b0023.png  
      inflating: /content/data/facades/test/b/cmp_b0026.png  
      inflating: /content/data/facades/test/b/cmp_b0028.png  
      inflating: /content/data/facades/test/b/cmp_b0034.png  
      inflating: /content/data/facades/test/b/cmp_b0035.png  
      inflating: /content/data/facades/test/b/cmp_b0047.png  
      inflating: /content/data/facades/test/b/cmp_b0049.png  
      inflating: /content/data/facades/test/b/cmp_b0054.png  
      inflating: /content/data/facades/test/b/cmp_b0055.png  
      inflating: /content/data/facades/test/b/cmp_b0057.png  
      inflating: /content/data/facades/test/b/cmp_b0061.png  
      inflating: /content/data/facades/test/b/cmp_b0068.png  
      inflating: /content/data/facades/test/b/cmp_b0071.png  
      inflating: /content/data/facades/test/b/cmp_b0073.png  
      inflating: /content/data/facades/test/b/cmp_b0075.png  
      inflating: /content/data/facades/test/b/cmp_b0077.png  
      inflating: /content/data/facades/test/b/cmp_b0078.png  
      inflating: /content/data/facades/test/b/cmp_b0081.png  
      inflating: /content/data/facades/test/b/cmp_b0084.png  
      inflating: /content/data/facades/test/b/cmp_b0085.png  
      inflating: /content/data/facades/test/b/cmp_b0087.png  
      inflating: /content/data/facades/test/b/cmp_b0089.png  
      inflating: /content/data/facades/test/b/cmp_b0090.png  
      inflating: /content/data/facades/test/b/cmp_b0093.png  
      inflating: /content/data/facades/test/b/cmp_b0095.png  
      inflating: /content/data/facades/test/b/cmp_b0096.png  
      inflating: /content/data/facades/test/b/cmp_b0103.png  
      inflating: /content/data/facades/test/b/cmp_b0105.png  
      inflating: /content/data/facades/test/b/cmp_b0112.png  
      inflating: /content/data/facades/test/b/cmp_b0113.png  
      inflating: /content/data/facades/test/b/cmp_b0121.png  
      inflating: /content/data/facades/test/b/cmp_b0122.png  
      inflating: /content/data/facades/test/b/cmp_b0124.png  
      inflating: /content/data/facades/test/b/cmp_b0125.png  
      inflating: /content/data/facades/test/b/cmp_b0126.png  
      inflating: /content/data/facades/test/b/cmp_b0127.png  
      inflating: /content/data/facades/test/b/cmp_b0135.png  
      inflating: /content/data/facades/test/b/cmp_b0141.png  
      inflating: /content/data/facades/test/b/cmp_b0145.png  
      inflating: /content/data/facades/test/b/cmp_b0146.png  
      inflating: /content/data/facades/test/b/cmp_b0148.png  
      inflating: /content/data/facades/test/b/cmp_b0149.png  
      inflating: /content/data/facades/test/b/cmp_b0150.png  
      inflating: /content/data/facades/test/b/cmp_b0154.png  
      inflating: /content/data/facades/test/b/cmp_b0156.png  
      inflating: /content/data/facades/test/b/cmp_b0157.png  
      inflating: /content/data/facades/test/b/cmp_b0162.png  
      inflating: /content/data/facades/test/b/cmp_b0164.png  
      inflating: /content/data/facades/test/b/cmp_b0165.png  
      inflating: /content/data/facades/test/b/cmp_b0166.png  
      inflating: /content/data/facades/test/b/cmp_b0167.png  
      inflating: /content/data/facades/test/b/cmp_b0168.png  
      inflating: /content/data/facades/test/b/cmp_b0172.png  
      inflating: /content/data/facades/test/b/cmp_b0174.png  
      inflating: /content/data/facades/test/b/cmp_b0184.png  
      inflating: /content/data/facades/test/b/cmp_b0185.png  
      inflating: /content/data/facades/test/b/cmp_b0187.png  
      inflating: /content/data/facades/test/b/cmp_b0188.png  
      inflating: /content/data/facades/test/b/cmp_b0190.png  
      inflating: /content/data/facades/test/b/cmp_b0192.png  
      inflating: /content/data/facades/test/b/cmp_b0199.png  
      inflating: /content/data/facades/test/b/cmp_b0200.png  
      inflating: /content/data/facades/test/b/cmp_b0202.png  
      inflating: /content/data/facades/test/b/cmp_b0204.png  
      inflating: /content/data/facades/test/b/cmp_b0206.png  
      inflating: /content/data/facades/test/b/cmp_b0207.png  
      inflating: /content/data/facades/test/b/cmp_b0214.png  
      inflating: /content/data/facades/test/b/cmp_b0218.png  
      inflating: /content/data/facades/test/b/cmp_b0219.png  
      inflating: /content/data/facades/test/b/cmp_b0227.png  
      inflating: /content/data/facades/test/b/cmp_b0231.png  
      inflating: /content/data/facades/test/b/cmp_b0233.png  
      inflating: /content/data/facades/test/b/cmp_b0234.png  
      inflating: /content/data/facades/test/b/cmp_b0235.png  
      inflating: /content/data/facades/test/b/cmp_b0239.png  
      inflating: /content/data/facades/test/b/cmp_b0241.png  
      inflating: /content/data/facades/test/b/cmp_b0242.png  
      inflating: /content/data/facades/test/b/cmp_b0246.png  
      inflating: /content/data/facades/test/b/cmp_b0247.png  
      inflating: /content/data/facades/test/b/cmp_b0250.png  
      inflating: /content/data/facades/test/b/cmp_b0258.png  
      inflating: /content/data/facades/test/b/cmp_b0260.png  
      inflating: /content/data/facades/test/b/cmp_b0271.png  
      inflating: /content/data/facades/test/b/cmp_b0272.png  
      inflating: /content/data/facades/test/b/cmp_b0275.png  
      inflating: /content/data/facades/test/b/cmp_b0280.png  
      inflating: /content/data/facades/test/b/cmp_b0283.png  
      inflating: /content/data/facades/test/b/cmp_b0285.png  
      inflating: /content/data/facades/test/b/cmp_b0286.png  
      inflating: /content/data/facades/test/b/cmp_b0288.png  
      inflating: /content/data/facades/test/b/cmp_b0290.png  
      inflating: /content/data/facades/test/b/cmp_b0291.png  
      inflating: /content/data/facades/test/b/cmp_b0292.png  
      inflating: /content/data/facades/test/b/cmp_b0293.png  
      inflating: /content/data/facades/test/b/cmp_b0294.png  
      inflating: /content/data/facades/test/b/cmp_b0298.png  
      inflating: /content/data/facades/test/b/cmp_b0300.png  
      inflating: /content/data/facades/test/b/cmp_b0302.png  
      inflating: /content/data/facades/test/b/cmp_b0305.png  
      inflating: /content/data/facades/test/b/cmp_b0306.png  
      inflating: /content/data/facades/test/b/cmp_b0313.png  
      inflating: /content/data/facades/test/b/cmp_b0315.png  
      inflating: /content/data/facades/test/b/cmp_b0319.png  
      inflating: /content/data/facades/test/b/cmp_b0325.png  
      inflating: /content/data/facades/test/b/cmp_b0331.png  
      inflating: /content/data/facades/test/b/cmp_b0334.png  
      inflating: /content/data/facades/test/b/cmp_b0340.png  
      inflating: /content/data/facades/test/b/cmp_b0341.png  
      inflating: /content/data/facades/test/b/cmp_b0343.png  
      inflating: /content/data/facades/test/b/cmp_b0345.png  
      inflating: /content/data/facades/test/b/cmp_b0349.png  
      inflating: /content/data/facades/test/b/cmp_b0352.png  
      inflating: /content/data/facades/test/b/cmp_b0353.png  
      inflating: /content/data/facades/test/b/cmp_b0355.png  
      inflating: /content/data/facades/test/b/cmp_b0360.png  
      inflating: /content/data/facades/test/b/cmp_b0363.png  
      inflating: /content/data/facades/test/b/cmp_b0364.png  
      inflating: /content/data/facades/test/b/cmp_b0365.png  
      inflating: /content/data/facades/test/b/cmp_b0368.png  
      inflating: /content/data/facades/test/b/cmp_b0370.png  
      inflating: /content/data/facades/test/b/cmp_b0372.png  
      inflating: /content/data/facades/test/b/cmp_b0373.png  
      inflating: /content/data/facades/test/b/cmp_b0376.png  
      inflating: /content/data/facades/test/b/cmp_b0377.png  
      inflating: /content/data/facades/test/b/cmp_x0003.png  
      inflating: /content/data/facades/test/b/cmp_x0006.png  
      inflating: /content/data/facades/test/b/cmp_x0007.png  
      inflating: /content/data/facades/test/b/cmp_x0009.png  
      inflating: /content/data/facades/test/b/cmp_x0011.png  
      inflating: /content/data/facades/test/b/cmp_x0013.png  
      inflating: /content/data/facades/test/b/cmp_x0023.png  
      inflating: /content/data/facades/test/b/cmp_x0024.png  
      inflating: /content/data/facades/test/b/cmp_x0026.png  
      inflating: /content/data/facades/test/b/cmp_x0027.png  
      inflating: /content/data/facades/test/b/cmp_x0032.png  
      inflating: /content/data/facades/test/b/cmp_x0033.png  
      inflating: /content/data/facades/test/b/cmp_x0039.png  
      inflating: /content/data/facades/test/b/cmp_x0040.png  
      inflating: /content/data/facades/test/b/cmp_x0041.png  
      inflating: /content/data/facades/test/b/cmp_x0046.png  
      inflating: /content/data/facades/test/b/cmp_x0047.png  
      inflating: /content/data/facades/test/b/cmp_x0053.png  
      inflating: /content/data/facades/test/b/cmp_x0057.png  
      inflating: /content/data/facades/test/b/cmp_x0059.png  
      inflating: /content/data/facades/test/b/cmp_x0060.png  
      inflating: /content/data/facades/test/b/cmp_x0070.png  
      inflating: /content/data/facades/test/b/cmp_x0073.png  
      inflating: /content/data/facades/test/b/cmp_x0076.png  
      inflating: /content/data/facades/test/b/cmp_x0077.png  
      inflating: /content/data/facades/test/b/cmp_x0080.png  
      inflating: /content/data/facades/test/b/cmp_x0083.png  
      inflating: /content/data/facades/test/b/cmp_x0084.png  
      inflating: /content/data/facades/test/b/cmp_x0086.png  
      inflating: /content/data/facades/test/b/cmp_x0093.png  
      inflating: /content/data/facades/test/b/cmp_x0096.png  
      inflating: /content/data/facades/test/b/cmp_x0097.png  
      inflating: /content/data/facades/test/b/cmp_x0100.png  
      inflating: /content/data/facades/test/b/cmp_x0101.png  
      inflating: /content/data/facades/test/b/cmp_x0105.png  
      inflating: /content/data/facades/test/b/cmp_x0107.png  
      inflating: /content/data/facades/test/b/cmp_x0114.png  
      inflating: /content/data/facades/test/b/cmp_x0116.png  
      inflating: /content/data/facades/test/b/cmp_x0117.png  
      inflating: /content/data/facades/test/b/cmp_x0119.png  
      inflating: /content/data/facades/test/b/cmp_x0126.png  
      inflating: /content/data/facades/test/b/cmp_x0127.png  
      inflating: /content/data/facades/test/b/cmp_x0128.png  
      inflating: /content/data/facades/test/b/cmp_x0129.png  
      inflating: /content/data/facades/test/b/cmp_x0130.png  
      inflating: /content/data/facades/test/b/cmp_x0133.png  
      inflating: /content/data/facades/test/b/cmp_x0142.png  
      inflating: /content/data/facades/test/b/cmp_x0143.png  
      inflating: /content/data/facades/test/b/cmp_x0147.png  
      inflating: /content/data/facades/test/b/cmp_x0151.png  
      inflating: /content/data/facades/test/b/cmp_x0156.png  
      inflating: /content/data/facades/test/b/cmp_x0157.png  
      inflating: /content/data/facades/test/b/cmp_x0159.png  
      inflating: /content/data/facades/test/b/cmp_x0160.png  
      inflating: /content/data/facades/test/b/cmp_x0161.png  
      inflating: /content/data/facades/test/b/cmp_x0163.png  
      inflating: /content/data/facades/test/b/cmp_x0164.png  
      inflating: /content/data/facades/test/b/cmp_x0169.png  
      inflating: /content/data/facades/test/b/cmp_x0177.png  
      inflating: /content/data/facades/test/b/cmp_x0183.png  
      inflating: /content/data/facades/test/b/cmp_x0184.png  
      inflating: /content/data/facades/test/b/cmp_x0186.png  
      inflating: /content/data/facades/test/b/cmp_x0188.png  
      inflating: /content/data/facades/test/b/cmp_x0197.png  
      inflating: /content/data/facades/test/b/cmp_x0198.png  
      inflating: /content/data/facades/test/b/cmp_x0205.png  
      inflating: /content/data/facades/test/b/cmp_x0207.png  
      inflating: /content/data/facades/test/b/cmp_x0210.png  
      inflating: /content/data/facades/test/b/cmp_x0213.png  
      inflating: /content/data/facades/test/b/cmp_x0214.png  
      inflating: /content/data/facades/test/b/cmp_x0216.png  
      inflating: /content/data/facades/test/b/cmp_x0217.png  
      inflating: /content/data/facades/test/b/cmp_x0220.png  
      inflating: /content/data/facades/test/b/cmp_x0221.png  
      inflating: /content/data/facades/test/b/cmp_x0223.png  
      inflating: /content/data/facades/test/b/cmp_x0224.png  
      inflating: /content/data/facades/test/b/cmp_x0226.png  
       creating: /content/data/facades/train/
       creating: /content/data/facades/train/a/
      inflating: /content/data/facades/train/a/cmp_b0005.png  
      inflating: /content/data/facades/train/a/cmp_b0006.png  
      inflating: /content/data/facades/train/a/cmp_b0007.png  
      inflating: /content/data/facades/train/a/cmp_b0008.png  
      inflating: /content/data/facades/train/a/cmp_b0009.png  
      inflating: /content/data/facades/train/a/cmp_b0010.png  
      inflating: /content/data/facades/train/a/cmp_b0011.png  
      inflating: /content/data/facades/train/a/cmp_b0012.png  
      inflating: /content/data/facades/train/a/cmp_b0014.png  
      inflating: /content/data/facades/train/a/cmp_b0015.png  
      inflating: /content/data/facades/train/a/cmp_b0016.png  
      inflating: /content/data/facades/train/a/cmp_b0017.png  
      inflating: /content/data/facades/train/a/cmp_b0018.png  
      inflating: /content/data/facades/train/a/cmp_b0019.png  
      inflating: /content/data/facades/train/a/cmp_b0020.png  
      inflating: /content/data/facades/train/a/cmp_b0021.png  
      inflating: /content/data/facades/train/a/cmp_b0022.png  
      inflating: /content/data/facades/train/a/cmp_b0024.png  
      inflating: /content/data/facades/train/a/cmp_b0025.png  
      inflating: /content/data/facades/train/a/cmp_b0027.png  
      inflating: /content/data/facades/train/a/cmp_b0029.png  
      inflating: /content/data/facades/train/a/cmp_b0030.png  
      inflating: /content/data/facades/train/a/cmp_b0031.png  
      inflating: /content/data/facades/train/a/cmp_b0032.png  
      inflating: /content/data/facades/train/a/cmp_b0033.png  
      inflating: /content/data/facades/train/a/cmp_b0036.png  
      inflating: /content/data/facades/train/a/cmp_b0037.png  
      inflating: /content/data/facades/train/a/cmp_b0038.png  
      inflating: /content/data/facades/train/a/cmp_b0039.png  
      inflating: /content/data/facades/train/a/cmp_b0040.png  
      inflating: /content/data/facades/train/a/cmp_b0041.png  
      inflating: /content/data/facades/train/a/cmp_b0042.png  
      inflating: /content/data/facades/train/a/cmp_b0043.png  
      inflating: /content/data/facades/train/a/cmp_b0044.png  
      inflating: /content/data/facades/train/a/cmp_b0045.png  
      inflating: /content/data/facades/train/a/cmp_b0046.png  
      inflating: /content/data/facades/train/a/cmp_b0048.png  
      inflating: /content/data/facades/train/a/cmp_b0050.png  
      inflating: /content/data/facades/train/a/cmp_b0051.png  
      inflating: /content/data/facades/train/a/cmp_b0052.png  
      inflating: /content/data/facades/train/a/cmp_b0053.png  
      inflating: /content/data/facades/train/a/cmp_b0056.png  
      inflating: /content/data/facades/train/a/cmp_b0058.png  
      inflating: /content/data/facades/train/a/cmp_b0059.png  
      inflating: /content/data/facades/train/a/cmp_b0060.png  
      inflating: /content/data/facades/train/a/cmp_b0062.png  
      inflating: /content/data/facades/train/a/cmp_b0063.png  
      inflating: /content/data/facades/train/a/cmp_b0064.png  
      inflating: /content/data/facades/train/a/cmp_b0065.png  
      inflating: /content/data/facades/train/a/cmp_b0066.png  
      inflating: /content/data/facades/train/a/cmp_b0067.png  
      inflating: /content/data/facades/train/a/cmp_b0069.png  
      inflating: /content/data/facades/train/a/cmp_b0070.png  
      inflating: /content/data/facades/train/a/cmp_b0072.png  
      inflating: /content/data/facades/train/a/cmp_b0074.png  
      inflating: /content/data/facades/train/a/cmp_b0076.png  
      inflating: /content/data/facades/train/a/cmp_b0079.png  
      inflating: /content/data/facades/train/a/cmp_b0080.png  
      inflating: /content/data/facades/train/a/cmp_b0082.png  
      inflating: /content/data/facades/train/a/cmp_b0083.png  
      inflating: /content/data/facades/train/a/cmp_b0086.png  
      inflating: /content/data/facades/train/a/cmp_b0088.png  
      inflating: /content/data/facades/train/a/cmp_b0091.png  
      inflating: /content/data/facades/train/a/cmp_b0092.png  
      inflating: /content/data/facades/train/a/cmp_b0094.png  
      inflating: /content/data/facades/train/a/cmp_b0097.png  
      inflating: /content/data/facades/train/a/cmp_b0098.png  
      inflating: /content/data/facades/train/a/cmp_b0099.png  
      inflating: /content/data/facades/train/a/cmp_b0100.png  
      inflating: /content/data/facades/train/a/cmp_b0101.png  
      inflating: /content/data/facades/train/a/cmp_b0102.png  
      inflating: /content/data/facades/train/a/cmp_b0104.png  
      inflating: /content/data/facades/train/a/cmp_b0106.png  
      inflating: /content/data/facades/train/a/cmp_b0107.png  
      inflating: /content/data/facades/train/a/cmp_b0108.png  
      inflating: /content/data/facades/train/a/cmp_b0109.png  
      inflating: /content/data/facades/train/a/cmp_b0110.png  
      inflating: /content/data/facades/train/a/cmp_b0111.png  
      inflating: /content/data/facades/train/a/cmp_b0114.png  
      inflating: /content/data/facades/train/a/cmp_b0115.png  
      inflating: /content/data/facades/train/a/cmp_b0116.png  
      inflating: /content/data/facades/train/a/cmp_b0117.png  
      inflating: /content/data/facades/train/a/cmp_b0118.png  
      inflating: /content/data/facades/train/a/cmp_b0119.png  
      inflating: /content/data/facades/train/a/cmp_b0120.png  
      inflating: /content/data/facades/train/a/cmp_b0123.png  
      inflating: /content/data/facades/train/a/cmp_b0128.png  
      inflating: /content/data/facades/train/a/cmp_b0129.png  
      inflating: /content/data/facades/train/a/cmp_b0130.png  
      inflating: /content/data/facades/train/a/cmp_b0131.png  
      inflating: /content/data/facades/train/a/cmp_b0132.png  
      inflating: /content/data/facades/train/a/cmp_b0133.png  
      inflating: /content/data/facades/train/a/cmp_b0134.png  
      inflating: /content/data/facades/train/a/cmp_b0136.png  
      inflating: /content/data/facades/train/a/cmp_b0137.png  
      inflating: /content/data/facades/train/a/cmp_b0138.png  
      inflating: /content/data/facades/train/a/cmp_b0139.png  
      inflating: /content/data/facades/train/a/cmp_b0140.png  
      inflating: /content/data/facades/train/a/cmp_b0142.png  
      inflating: /content/data/facades/train/a/cmp_b0143.png  
      inflating: /content/data/facades/train/a/cmp_b0144.png  
      inflating: /content/data/facades/train/a/cmp_b0147.png  
      inflating: /content/data/facades/train/a/cmp_b0151.png  
      inflating: /content/data/facades/train/a/cmp_b0152.png  
      inflating: /content/data/facades/train/a/cmp_b0153.png  
      inflating: /content/data/facades/train/a/cmp_b0155.png  
      inflating: /content/data/facades/train/a/cmp_b0158.png  
      inflating: /content/data/facades/train/a/cmp_b0159.png  
      inflating: /content/data/facades/train/a/cmp_b0160.png  
      inflating: /content/data/facades/train/a/cmp_b0161.png  
      inflating: /content/data/facades/train/a/cmp_b0163.png  
      inflating: /content/data/facades/train/a/cmp_b0169.png  
      inflating: /content/data/facades/train/a/cmp_b0170.png  
      inflating: /content/data/facades/train/a/cmp_b0171.png  
      inflating: /content/data/facades/train/a/cmp_b0173.png  
      inflating: /content/data/facades/train/a/cmp_b0175.png  
      inflating: /content/data/facades/train/a/cmp_b0176.png  
      inflating: /content/data/facades/train/a/cmp_b0177.png  
      inflating: /content/data/facades/train/a/cmp_b0178.png  
      inflating: /content/data/facades/train/a/cmp_b0179.png  
      inflating: /content/data/facades/train/a/cmp_b0180.png  
      inflating: /content/data/facades/train/a/cmp_b0181.png  
      inflating: /content/data/facades/train/a/cmp_b0182.png  
      inflating: /content/data/facades/train/a/cmp_b0183.png  
      inflating: /content/data/facades/train/a/cmp_b0186.png  
      inflating: /content/data/facades/train/a/cmp_b0189.png  
      inflating: /content/data/facades/train/a/cmp_b0191.png  
      inflating: /content/data/facades/train/a/cmp_b0193.png  
      inflating: /content/data/facades/train/a/cmp_b0194.png  
      inflating: /content/data/facades/train/a/cmp_b0195.png  
      inflating: /content/data/facades/train/a/cmp_b0196.png  
      inflating: /content/data/facades/train/a/cmp_b0197.png  
      inflating: /content/data/facades/train/a/cmp_b0198.png  
      inflating: /content/data/facades/train/a/cmp_b0201.png  
      inflating: /content/data/facades/train/a/cmp_b0203.png  
      inflating: /content/data/facades/train/a/cmp_b0205.png  
      inflating: /content/data/facades/train/a/cmp_b0208.png  
      inflating: /content/data/facades/train/a/cmp_b0209.png  
      inflating: /content/data/facades/train/a/cmp_b0210.png  
      inflating: /content/data/facades/train/a/cmp_b0211.png  
      inflating: /content/data/facades/train/a/cmp_b0212.png  
      inflating: /content/data/facades/train/a/cmp_b0213.png  
      inflating: /content/data/facades/train/a/cmp_b0215.png  
      inflating: /content/data/facades/train/a/cmp_b0216.png  
      inflating: /content/data/facades/train/a/cmp_b0217.png  
      inflating: /content/data/facades/train/a/cmp_b0220.png  
      inflating: /content/data/facades/train/a/cmp_b0221.png  
      inflating: /content/data/facades/train/a/cmp_b0222.png  
      inflating: /content/data/facades/train/a/cmp_b0223.png  
      inflating: /content/data/facades/train/a/cmp_b0224.png  
      inflating: /content/data/facades/train/a/cmp_b0225.png  
      inflating: /content/data/facades/train/a/cmp_b0226.png  
      inflating: /content/data/facades/train/a/cmp_b0228.png  
      inflating: /content/data/facades/train/a/cmp_b0229.png  
      inflating: /content/data/facades/train/a/cmp_b0230.png  
      inflating: /content/data/facades/train/a/cmp_b0232.png  
      inflating: /content/data/facades/train/a/cmp_b0236.png  
      inflating: /content/data/facades/train/a/cmp_b0237.png  
      inflating: /content/data/facades/train/a/cmp_b0238.png  
      inflating: /content/data/facades/train/a/cmp_b0240.png  
      inflating: /content/data/facades/train/a/cmp_b0243.png  
      inflating: /content/data/facades/train/a/cmp_b0244.png  
      inflating: /content/data/facades/train/a/cmp_b0245.png  
      inflating: /content/data/facades/train/a/cmp_b0248.png  
      inflating: /content/data/facades/train/a/cmp_b0249.png  
      inflating: /content/data/facades/train/a/cmp_b0251.png  
      inflating: /content/data/facades/train/a/cmp_b0252.png  
      inflating: /content/data/facades/train/a/cmp_b0253.png  
      inflating: /content/data/facades/train/a/cmp_b0254.png  
      inflating: /content/data/facades/train/a/cmp_b0255.png  
      inflating: /content/data/facades/train/a/cmp_b0256.png  
      inflating: /content/data/facades/train/a/cmp_b0257.png  
      inflating: /content/data/facades/train/a/cmp_b0259.png  
      inflating: /content/data/facades/train/a/cmp_b0261.png  
      inflating: /content/data/facades/train/a/cmp_b0262.png  
      inflating: /content/data/facades/train/a/cmp_b0263.png  
      inflating: /content/data/facades/train/a/cmp_b0264.png  
      inflating: /content/data/facades/train/a/cmp_b0265.png  
      inflating: /content/data/facades/train/a/cmp_b0266.png  
      inflating: /content/data/facades/train/a/cmp_b0267.png  
      inflating: /content/data/facades/train/a/cmp_b0268.png  
      inflating: /content/data/facades/train/a/cmp_b0269.png  
      inflating: /content/data/facades/train/a/cmp_b0270.png  
      inflating: /content/data/facades/train/a/cmp_b0273.png  
      inflating: /content/data/facades/train/a/cmp_b0274.png  
      inflating: /content/data/facades/train/a/cmp_b0276.png  
      inflating: /content/data/facades/train/a/cmp_b0277.png  
      inflating: /content/data/facades/train/a/cmp_b0278.png  
      inflating: /content/data/facades/train/a/cmp_b0279.png  
      inflating: /content/data/facades/train/a/cmp_b0281.png  
      inflating: /content/data/facades/train/a/cmp_b0282.png  
      inflating: /content/data/facades/train/a/cmp_b0284.png  
      inflating: /content/data/facades/train/a/cmp_b0287.png  
      inflating: /content/data/facades/train/a/cmp_b0289.png  
      inflating: /content/data/facades/train/a/cmp_b0295.png  
      inflating: /content/data/facades/train/a/cmp_b0296.png  
      inflating: /content/data/facades/train/a/cmp_b0297.png  
      inflating: /content/data/facades/train/a/cmp_b0299.png  
      inflating: /content/data/facades/train/a/cmp_b0301.png  
      inflating: /content/data/facades/train/a/cmp_b0303.png  
      inflating: /content/data/facades/train/a/cmp_b0304.png  
      inflating: /content/data/facades/train/a/cmp_b0307.png  
      inflating: /content/data/facades/train/a/cmp_b0308.png  
      inflating: /content/data/facades/train/a/cmp_b0309.png  
      inflating: /content/data/facades/train/a/cmp_b0310.png  
      inflating: /content/data/facades/train/a/cmp_b0311.png  
      inflating: /content/data/facades/train/a/cmp_b0312.png  
      inflating: /content/data/facades/train/a/cmp_b0314.png  
      inflating: /content/data/facades/train/a/cmp_b0316.png  
      inflating: /content/data/facades/train/a/cmp_b0317.png  
      inflating: /content/data/facades/train/a/cmp_b0318.png  
      inflating: /content/data/facades/train/a/cmp_b0320.png  
      inflating: /content/data/facades/train/a/cmp_b0321.png  
      inflating: /content/data/facades/train/a/cmp_b0322.png  
      inflating: /content/data/facades/train/a/cmp_b0323.png  
      inflating: /content/data/facades/train/a/cmp_b0324.png  
      inflating: /content/data/facades/train/a/cmp_b0326.png  
      inflating: /content/data/facades/train/a/cmp_b0327.png  
      inflating: /content/data/facades/train/a/cmp_b0328.png  
      inflating: /content/data/facades/train/a/cmp_b0329.png  
      inflating: /content/data/facades/train/a/cmp_b0330.png  
      inflating: /content/data/facades/train/a/cmp_b0332.png  
      inflating: /content/data/facades/train/a/cmp_b0333.png  
      inflating: /content/data/facades/train/a/cmp_b0335.png  
      inflating: /content/data/facades/train/a/cmp_b0336.png  
      inflating: /content/data/facades/train/a/cmp_b0337.png  
      inflating: /content/data/facades/train/a/cmp_b0338.png  
      inflating: /content/data/facades/train/a/cmp_b0339.png  
      inflating: /content/data/facades/train/a/cmp_b0342.png  
      inflating: /content/data/facades/train/a/cmp_b0344.png  
      inflating: /content/data/facades/train/a/cmp_b0346.png  
      inflating: /content/data/facades/train/a/cmp_b0347.png  
      inflating: /content/data/facades/train/a/cmp_b0348.png  
      inflating: /content/data/facades/train/a/cmp_b0350.png  
      inflating: /content/data/facades/train/a/cmp_b0351.png  
      inflating: /content/data/facades/train/a/cmp_b0354.png  
      inflating: /content/data/facades/train/a/cmp_b0356.png  
      inflating: /content/data/facades/train/a/cmp_b0357.png  
      inflating: /content/data/facades/train/a/cmp_b0358.png  
      inflating: /content/data/facades/train/a/cmp_b0359.png  
      inflating: /content/data/facades/train/a/cmp_b0361.png  
      inflating: /content/data/facades/train/a/cmp_b0362.png  
      inflating: /content/data/facades/train/a/cmp_b0366.png  
      inflating: /content/data/facades/train/a/cmp_b0367.png  
      inflating: /content/data/facades/train/a/cmp_b0369.png  
      inflating: /content/data/facades/train/a/cmp_b0371.png  
      inflating: /content/data/facades/train/a/cmp_b0374.png  
      inflating: /content/data/facades/train/a/cmp_b0375.png  
      inflating: /content/data/facades/train/a/cmp_b0378.png  
      inflating: /content/data/facades/train/a/cmp_x0001.png  
      inflating: /content/data/facades/train/a/cmp_x0002.png  
      inflating: /content/data/facades/train/a/cmp_x0004.png  
      inflating: /content/data/facades/train/a/cmp_x0005.png  
      inflating: /content/data/facades/train/a/cmp_x0008.png  
      inflating: /content/data/facades/train/a/cmp_x0010.png  
      inflating: /content/data/facades/train/a/cmp_x0012.png  
      inflating: /content/data/facades/train/a/cmp_x0014.png  
      inflating: /content/data/facades/train/a/cmp_x0015.png  
      inflating: /content/data/facades/train/a/cmp_x0016.png  
      inflating: /content/data/facades/train/a/cmp_x0017.png  
      inflating: /content/data/facades/train/a/cmp_x0018.png  
      inflating: /content/data/facades/train/a/cmp_x0019.png  
      inflating: /content/data/facades/train/a/cmp_x0020.png  
      inflating: /content/data/facades/train/a/cmp_x0021.png  
      inflating: /content/data/facades/train/a/cmp_x0022.png  
      inflating: /content/data/facades/train/a/cmp_x0025.png  
      inflating: /content/data/facades/train/a/cmp_x0028.png  
      inflating: /content/data/facades/train/a/cmp_x0029.png  
      inflating: /content/data/facades/train/a/cmp_x0030.png  
      inflating: /content/data/facades/train/a/cmp_x0031.png  
      inflating: /content/data/facades/train/a/cmp_x0034.png  
      inflating: /content/data/facades/train/a/cmp_x0035.png  
      inflating: /content/data/facades/train/a/cmp_x0036.png  
      inflating: /content/data/facades/train/a/cmp_x0037.png  
      inflating: /content/data/facades/train/a/cmp_x0038.png  
      inflating: /content/data/facades/train/a/cmp_x0042.png  
      inflating: /content/data/facades/train/a/cmp_x0043.png  
      inflating: /content/data/facades/train/a/cmp_x0044.png  
      inflating: /content/data/facades/train/a/cmp_x0045.png  
      inflating: /content/data/facades/train/a/cmp_x0048.png  
      inflating: /content/data/facades/train/a/cmp_x0049.png  
      inflating: /content/data/facades/train/a/cmp_x0050.png  
      inflating: /content/data/facades/train/a/cmp_x0051.png  
      inflating: /content/data/facades/train/a/cmp_x0052.png  
      inflating: /content/data/facades/train/a/cmp_x0054.png  
      inflating: /content/data/facades/train/a/cmp_x0055.png  
      inflating: /content/data/facades/train/a/cmp_x0056.png  
      inflating: /content/data/facades/train/a/cmp_x0058.png  
      inflating: /content/data/facades/train/a/cmp_x0061.png  
      inflating: /content/data/facades/train/a/cmp_x0062.png  
      inflating: /content/data/facades/train/a/cmp_x0063.png  
      inflating: /content/data/facades/train/a/cmp_x0064.png  
      inflating: /content/data/facades/train/a/cmp_x0065.png  
      inflating: /content/data/facades/train/a/cmp_x0066.png  
      inflating: /content/data/facades/train/a/cmp_x0067.png  
      inflating: /content/data/facades/train/a/cmp_x0068.png  
      inflating: /content/data/facades/train/a/cmp_x0069.png  
      inflating: /content/data/facades/train/a/cmp_x0071.png  
      inflating: /content/data/facades/train/a/cmp_x0072.png  
      inflating: /content/data/facades/train/a/cmp_x0074.png  
      inflating: /content/data/facades/train/a/cmp_x0075.png  
      inflating: /content/data/facades/train/a/cmp_x0078.png  
      inflating: /content/data/facades/train/a/cmp_x0079.png  
      inflating: /content/data/facades/train/a/cmp_x0081.png  
      inflating: /content/data/facades/train/a/cmp_x0082.png  
      inflating: /content/data/facades/train/a/cmp_x0085.png  
      inflating: /content/data/facades/train/a/cmp_x0087.png  
      inflating: /content/data/facades/train/a/cmp_x0088.png  
      inflating: /content/data/facades/train/a/cmp_x0089.png  
      inflating: /content/data/facades/train/a/cmp_x0090.png  
      inflating: /content/data/facades/train/a/cmp_x0091.png  
      inflating: /content/data/facades/train/a/cmp_x0092.png  
      inflating: /content/data/facades/train/a/cmp_x0094.png  
      inflating: /content/data/facades/train/a/cmp_x0095.png  
      inflating: /content/data/facades/train/a/cmp_x0098.png  
      inflating: /content/data/facades/train/a/cmp_x0099.png  
      inflating: /content/data/facades/train/a/cmp_x0102.png  
      inflating: /content/data/facades/train/a/cmp_x0103.png  
      inflating: /content/data/facades/train/a/cmp_x0104.png  
      inflating: /content/data/facades/train/a/cmp_x0106.png  
      inflating: /content/data/facades/train/a/cmp_x0108.png  
      inflating: /content/data/facades/train/a/cmp_x0109.png  
      inflating: /content/data/facades/train/a/cmp_x0110.png  
      inflating: /content/data/facades/train/a/cmp_x0111.png  
      inflating: /content/data/facades/train/a/cmp_x0112.png  
      inflating: /content/data/facades/train/a/cmp_x0113.png  
      inflating: /content/data/facades/train/a/cmp_x0115.png  
      inflating: /content/data/facades/train/a/cmp_x0118.png  
      inflating: /content/data/facades/train/a/cmp_x0120.png  
      inflating: /content/data/facades/train/a/cmp_x0121.png  
      inflating: /content/data/facades/train/a/cmp_x0122.png  
      inflating: /content/data/facades/train/a/cmp_x0123.png  
      inflating: /content/data/facades/train/a/cmp_x0124.png  
      inflating: /content/data/facades/train/a/cmp_x0125.png  
      inflating: /content/data/facades/train/a/cmp_x0131.png  
      inflating: /content/data/facades/train/a/cmp_x0132.png  
      inflating: /content/data/facades/train/a/cmp_x0134.png  
      inflating: /content/data/facades/train/a/cmp_x0135.png  
      inflating: /content/data/facades/train/a/cmp_x0136.png  
      inflating: /content/data/facades/train/a/cmp_x0137.png  
      inflating: /content/data/facades/train/a/cmp_x0138.png  
      inflating: /content/data/facades/train/a/cmp_x0139.png  
      inflating: /content/data/facades/train/a/cmp_x0140.png  
      inflating: /content/data/facades/train/a/cmp_x0141.png  
      inflating: /content/data/facades/train/a/cmp_x0144.png  
      inflating: /content/data/facades/train/a/cmp_x0145.png  
      inflating: /content/data/facades/train/a/cmp_x0146.png  
      inflating: /content/data/facades/train/a/cmp_x0148.png  
      inflating: /content/data/facades/train/a/cmp_x0149.png  
      inflating: /content/data/facades/train/a/cmp_x0150.png  
      inflating: /content/data/facades/train/a/cmp_x0152.png  
      inflating: /content/data/facades/train/a/cmp_x0153.png  
      inflating: /content/data/facades/train/a/cmp_x0154.png  
      inflating: /content/data/facades/train/a/cmp_x0155.png  
      inflating: /content/data/facades/train/a/cmp_x0158.png  
      inflating: /content/data/facades/train/a/cmp_x0162.png  
      inflating: /content/data/facades/train/a/cmp_x0165.png  
      inflating: /content/data/facades/train/a/cmp_x0166.png  
      inflating: /content/data/facades/train/a/cmp_x0167.png  
      inflating: /content/data/facades/train/a/cmp_x0168.png  
      inflating: /content/data/facades/train/a/cmp_x0170.png  
      inflating: /content/data/facades/train/a/cmp_x0171.png  
      inflating: /content/data/facades/train/a/cmp_x0172.png  
      inflating: /content/data/facades/train/a/cmp_x0173.png  
      inflating: /content/data/facades/train/a/cmp_x0174.png  
      inflating: /content/data/facades/train/a/cmp_x0175.png  
      inflating: /content/data/facades/train/a/cmp_x0176.png  
      inflating: /content/data/facades/train/a/cmp_x0178.png  
      inflating: /content/data/facades/train/a/cmp_x0179.png  
      inflating: /content/data/facades/train/a/cmp_x0180.png  
      inflating: /content/data/facades/train/a/cmp_x0181.png  
      inflating: /content/data/facades/train/a/cmp_x0182.png  
      inflating: /content/data/facades/train/a/cmp_x0185.png  
      inflating: /content/data/facades/train/a/cmp_x0187.png  
      inflating: /content/data/facades/train/a/cmp_x0189.png  
      inflating: /content/data/facades/train/a/cmp_x0190.png  
      inflating: /content/data/facades/train/a/cmp_x0191.png  
      inflating: /content/data/facades/train/a/cmp_x0192.png  
      inflating: /content/data/facades/train/a/cmp_x0193.png  
      inflating: /content/data/facades/train/a/cmp_x0194.png  
      inflating: /content/data/facades/train/a/cmp_x0195.png  
      inflating: /content/data/facades/train/a/cmp_x0196.png  
      inflating: /content/data/facades/train/a/cmp_x0199.png  
      inflating: /content/data/facades/train/a/cmp_x0200.png  
      inflating: /content/data/facades/train/a/cmp_x0201.png  
      inflating: /content/data/facades/train/a/cmp_x0202.png  
      inflating: /content/data/facades/train/a/cmp_x0203.png  
      inflating: /content/data/facades/train/a/cmp_x0204.png  
      inflating: /content/data/facades/train/a/cmp_x0206.png  
      inflating: /content/data/facades/train/a/cmp_x0208.png  
      inflating: /content/data/facades/train/a/cmp_x0209.png  
      inflating: /content/data/facades/train/a/cmp_x0211.png  
      inflating: /content/data/facades/train/a/cmp_x0212.png  
      inflating: /content/data/facades/train/a/cmp_x0215.png  
      inflating: /content/data/facades/train/a/cmp_x0218.png  
      inflating: /content/data/facades/train/a/cmp_x0219.png  
      inflating: /content/data/facades/train/a/cmp_x0222.png  
      inflating: /content/data/facades/train/a/cmp_x0225.png  
      inflating: /content/data/facades/train/a/cmp_x0227.png  
      inflating: /content/data/facades/train/a/cmp_x0228.png  
       creating: /content/data/facades/train/b/
      inflating: /content/data/facades/train/b/cmp_b0005.png  
      inflating: /content/data/facades/train/b/cmp_b0006.png  
      inflating: /content/data/facades/train/b/cmp_b0007.png  
      inflating: /content/data/facades/train/b/cmp_b0008.png  
      inflating: /content/data/facades/train/b/cmp_b0009.png  
      inflating: /content/data/facades/train/b/cmp_b0010.png  
      inflating: /content/data/facades/train/b/cmp_b0011.png  
      inflating: /content/data/facades/train/b/cmp_b0012.png  
      inflating: /content/data/facades/train/b/cmp_b0014.png  
      inflating: /content/data/facades/train/b/cmp_b0015.png  
      inflating: /content/data/facades/train/b/cmp_b0016.png  
      inflating: /content/data/facades/train/b/cmp_b0017.png  
      inflating: /content/data/facades/train/b/cmp_b0018.png  
      inflating: /content/data/facades/train/b/cmp_b0019.png  
      inflating: /content/data/facades/train/b/cmp_b0020.png  
      inflating: /content/data/facades/train/b/cmp_b0021.png  
      inflating: /content/data/facades/train/b/cmp_b0022.png  
      inflating: /content/data/facades/train/b/cmp_b0024.png  
      inflating: /content/data/facades/train/b/cmp_b0025.png  
      inflating: /content/data/facades/train/b/cmp_b0027.png  
      inflating: /content/data/facades/train/b/cmp_b0029.png  
      inflating: /content/data/facades/train/b/cmp_b0030.png  
      inflating: /content/data/facades/train/b/cmp_b0031.png  
      inflating: /content/data/facades/train/b/cmp_b0032.png  
      inflating: /content/data/facades/train/b/cmp_b0033.png  
      inflating: /content/data/facades/train/b/cmp_b0036.png  
      inflating: /content/data/facades/train/b/cmp_b0037.png  
      inflating: /content/data/facades/train/b/cmp_b0038.png  
      inflating: /content/data/facades/train/b/cmp_b0039.png  
      inflating: /content/data/facades/train/b/cmp_b0040.png  
      inflating: /content/data/facades/train/b/cmp_b0041.png  
      inflating: /content/data/facades/train/b/cmp_b0042.png  
      inflating: /content/data/facades/train/b/cmp_b0043.png  
      inflating: /content/data/facades/train/b/cmp_b0044.png  
      inflating: /content/data/facades/train/b/cmp_b0045.png  
      inflating: /content/data/facades/train/b/cmp_b0046.png  
      inflating: /content/data/facades/train/b/cmp_b0048.png  
      inflating: /content/data/facades/train/b/cmp_b0050.png  
      inflating: /content/data/facades/train/b/cmp_b0051.png  
      inflating: /content/data/facades/train/b/cmp_b0052.png  
      inflating: /content/data/facades/train/b/cmp_b0053.png  
      inflating: /content/data/facades/train/b/cmp_b0056.png  
      inflating: /content/data/facades/train/b/cmp_b0058.png  
      inflating: /content/data/facades/train/b/cmp_b0059.png  
      inflating: /content/data/facades/train/b/cmp_b0060.png  
      inflating: /content/data/facades/train/b/cmp_b0062.png  
      inflating: /content/data/facades/train/b/cmp_b0063.png  
      inflating: /content/data/facades/train/b/cmp_b0064.png  
      inflating: /content/data/facades/train/b/cmp_b0065.png  
      inflating: /content/data/facades/train/b/cmp_b0066.png  
      inflating: /content/data/facades/train/b/cmp_b0067.png  
      inflating: /content/data/facades/train/b/cmp_b0069.png  
      inflating: /content/data/facades/train/b/cmp_b0070.png  
      inflating: /content/data/facades/train/b/cmp_b0072.png  
      inflating: /content/data/facades/train/b/cmp_b0074.png  
      inflating: /content/data/facades/train/b/cmp_b0076.png  
      inflating: /content/data/facades/train/b/cmp_b0079.png  
      inflating: /content/data/facades/train/b/cmp_b0080.png  
      inflating: /content/data/facades/train/b/cmp_b0082.png  
      inflating: /content/data/facades/train/b/cmp_b0083.png  
      inflating: /content/data/facades/train/b/cmp_b0086.png  
      inflating: /content/data/facades/train/b/cmp_b0088.png  
      inflating: /content/data/facades/train/b/cmp_b0091.png  
      inflating: /content/data/facades/train/b/cmp_b0092.png  
      inflating: /content/data/facades/train/b/cmp_b0094.png  
      inflating: /content/data/facades/train/b/cmp_b0097.png  
      inflating: /content/data/facades/train/b/cmp_b0098.png  
      inflating: /content/data/facades/train/b/cmp_b0099.png  
      inflating: /content/data/facades/train/b/cmp_b0100.png  
      inflating: /content/data/facades/train/b/cmp_b0101.png  
      inflating: /content/data/facades/train/b/cmp_b0102.png  
      inflating: /content/data/facades/train/b/cmp_b0104.png  
      inflating: /content/data/facades/train/b/cmp_b0106.png  
      inflating: /content/data/facades/train/b/cmp_b0107.png  
      inflating: /content/data/facades/train/b/cmp_b0108.png  
      inflating: /content/data/facades/train/b/cmp_b0109.png  
      inflating: /content/data/facades/train/b/cmp_b0110.png  
      inflating: /content/data/facades/train/b/cmp_b0111.png  
      inflating: /content/data/facades/train/b/cmp_b0114.png  
      inflating: /content/data/facades/train/b/cmp_b0115.png  
      inflating: /content/data/facades/train/b/cmp_b0116.png  
      inflating: /content/data/facades/train/b/cmp_b0117.png  
      inflating: /content/data/facades/train/b/cmp_b0118.png  
      inflating: /content/data/facades/train/b/cmp_b0119.png  
      inflating: /content/data/facades/train/b/cmp_b0120.png  
      inflating: /content/data/facades/train/b/cmp_b0123.png  
      inflating: /content/data/facades/train/b/cmp_b0128.png  
      inflating: /content/data/facades/train/b/cmp_b0129.png  
      inflating: /content/data/facades/train/b/cmp_b0130.png  
      inflating: /content/data/facades/train/b/cmp_b0131.png  
      inflating: /content/data/facades/train/b/cmp_b0132.png  
      inflating: /content/data/facades/train/b/cmp_b0133.png  
      inflating: /content/data/facades/train/b/cmp_b0134.png  
      inflating: /content/data/facades/train/b/cmp_b0136.png  
      inflating: /content/data/facades/train/b/cmp_b0137.png  
      inflating: /content/data/facades/train/b/cmp_b0138.png  
      inflating: /content/data/facades/train/b/cmp_b0139.png  
      inflating: /content/data/facades/train/b/cmp_b0140.png  
      inflating: /content/data/facades/train/b/cmp_b0142.png  
      inflating: /content/data/facades/train/b/cmp_b0143.png  
      inflating: /content/data/facades/train/b/cmp_b0144.png  
      inflating: /content/data/facades/train/b/cmp_b0147.png  
      inflating: /content/data/facades/train/b/cmp_b0151.png  
      inflating: /content/data/facades/train/b/cmp_b0152.png  
      inflating: /content/data/facades/train/b/cmp_b0153.png  
      inflating: /content/data/facades/train/b/cmp_b0155.png  
      inflating: /content/data/facades/train/b/cmp_b0158.png  
      inflating: /content/data/facades/train/b/cmp_b0159.png  
      inflating: /content/data/facades/train/b/cmp_b0160.png  
      inflating: /content/data/facades/train/b/cmp_b0161.png  
      inflating: /content/data/facades/train/b/cmp_b0163.png  
      inflating: /content/data/facades/train/b/cmp_b0169.png  
      inflating: /content/data/facades/train/b/cmp_b0170.png  
      inflating: /content/data/facades/train/b/cmp_b0171.png  
      inflating: /content/data/facades/train/b/cmp_b0173.png  
      inflating: /content/data/facades/train/b/cmp_b0175.png  
      inflating: /content/data/facades/train/b/cmp_b0176.png  
      inflating: /content/data/facades/train/b/cmp_b0177.png  
      inflating: /content/data/facades/train/b/cmp_b0178.png  
      inflating: /content/data/facades/train/b/cmp_b0179.png  
      inflating: /content/data/facades/train/b/cmp_b0180.png  
      inflating: /content/data/facades/train/b/cmp_b0181.png  
      inflating: /content/data/facades/train/b/cmp_b0182.png  
      inflating: /content/data/facades/train/b/cmp_b0183.png  
      inflating: /content/data/facades/train/b/cmp_b0186.png  
      inflating: /content/data/facades/train/b/cmp_b0189.png  
      inflating: /content/data/facades/train/b/cmp_b0191.png  
      inflating: /content/data/facades/train/b/cmp_b0193.png  
      inflating: /content/data/facades/train/b/cmp_b0194.png  
      inflating: /content/data/facades/train/b/cmp_b0195.png  
      inflating: /content/data/facades/train/b/cmp_b0196.png  
      inflating: /content/data/facades/train/b/cmp_b0197.png  
      inflating: /content/data/facades/train/b/cmp_b0198.png  
      inflating: /content/data/facades/train/b/cmp_b0201.png  
      inflating: /content/data/facades/train/b/cmp_b0203.png  
      inflating: /content/data/facades/train/b/cmp_b0205.png  
      inflating: /content/data/facades/train/b/cmp_b0208.png  
      inflating: /content/data/facades/train/b/cmp_b0209.png  
      inflating: /content/data/facades/train/b/cmp_b0210.png  
      inflating: /content/data/facades/train/b/cmp_b0211.png  
      inflating: /content/data/facades/train/b/cmp_b0212.png  
      inflating: /content/data/facades/train/b/cmp_b0213.png  
      inflating: /content/data/facades/train/b/cmp_b0215.png  
      inflating: /content/data/facades/train/b/cmp_b0216.png  
      inflating: /content/data/facades/train/b/cmp_b0217.png  
      inflating: /content/data/facades/train/b/cmp_b0220.png  
      inflating: /content/data/facades/train/b/cmp_b0221.png  
      inflating: /content/data/facades/train/b/cmp_b0222.png  
      inflating: /content/data/facades/train/b/cmp_b0223.png  
      inflating: /content/data/facades/train/b/cmp_b0224.png  
      inflating: /content/data/facades/train/b/cmp_b0225.png  
      inflating: /content/data/facades/train/b/cmp_b0226.png  
      inflating: /content/data/facades/train/b/cmp_b0228.png  
      inflating: /content/data/facades/train/b/cmp_b0229.png  
      inflating: /content/data/facades/train/b/cmp_b0230.png  
      inflating: /content/data/facades/train/b/cmp_b0232.png  
      inflating: /content/data/facades/train/b/cmp_b0236.png  
      inflating: /content/data/facades/train/b/cmp_b0237.png  
      inflating: /content/data/facades/train/b/cmp_b0238.png  
      inflating: /content/data/facades/train/b/cmp_b0240.png  
      inflating: /content/data/facades/train/b/cmp_b0243.png  
      inflating: /content/data/facades/train/b/cmp_b0244.png  
      inflating: /content/data/facades/train/b/cmp_b0245.png  
      inflating: /content/data/facades/train/b/cmp_b0248.png  
      inflating: /content/data/facades/train/b/cmp_b0249.png  
      inflating: /content/data/facades/train/b/cmp_b0251.png  
      inflating: /content/data/facades/train/b/cmp_b0252.png  
      inflating: /content/data/facades/train/b/cmp_b0253.png  
      inflating: /content/data/facades/train/b/cmp_b0254.png  
      inflating: /content/data/facades/train/b/cmp_b0255.png  
      inflating: /content/data/facades/train/b/cmp_b0256.png  
      inflating: /content/data/facades/train/b/cmp_b0257.png  
      inflating: /content/data/facades/train/b/cmp_b0259.png  
      inflating: /content/data/facades/train/b/cmp_b0261.png  
      inflating: /content/data/facades/train/b/cmp_b0262.png  
      inflating: /content/data/facades/train/b/cmp_b0263.png  
      inflating: /content/data/facades/train/b/cmp_b0264.png  
      inflating: /content/data/facades/train/b/cmp_b0265.png  
      inflating: /content/data/facades/train/b/cmp_b0266.png  
      inflating: /content/data/facades/train/b/cmp_b0267.png  
      inflating: /content/data/facades/train/b/cmp_b0268.png  
      inflating: /content/data/facades/train/b/cmp_b0269.png  
      inflating: /content/data/facades/train/b/cmp_b0270.png  
      inflating: /content/data/facades/train/b/cmp_b0273.png  
      inflating: /content/data/facades/train/b/cmp_b0274.png  
      inflating: /content/data/facades/train/b/cmp_b0276.png  
      inflating: /content/data/facades/train/b/cmp_b0277.png  
      inflating: /content/data/facades/train/b/cmp_b0278.png  
      inflating: /content/data/facades/train/b/cmp_b0279.png  
      inflating: /content/data/facades/train/b/cmp_b0281.png  
      inflating: /content/data/facades/train/b/cmp_b0282.png  
      inflating: /content/data/facades/train/b/cmp_b0284.png  
      inflating: /content/data/facades/train/b/cmp_b0287.png  
      inflating: /content/data/facades/train/b/cmp_b0289.png  
      inflating: /content/data/facades/train/b/cmp_b0295.png  
      inflating: /content/data/facades/train/b/cmp_b0296.png  
      inflating: /content/data/facades/train/b/cmp_b0297.png  
      inflating: /content/data/facades/train/b/cmp_b0299.png  
      inflating: /content/data/facades/train/b/cmp_b0301.png  
      inflating: /content/data/facades/train/b/cmp_b0303.png  
      inflating: /content/data/facades/train/b/cmp_b0304.png  
      inflating: /content/data/facades/train/b/cmp_b0307.png  
      inflating: /content/data/facades/train/b/cmp_b0308.png  
      inflating: /content/data/facades/train/b/cmp_b0309.png  
      inflating: /content/data/facades/train/b/cmp_b0310.png  
      inflating: /content/data/facades/train/b/cmp_b0311.png  
      inflating: /content/data/facades/train/b/cmp_b0312.png  
      inflating: /content/data/facades/train/b/cmp_b0314.png  
      inflating: /content/data/facades/train/b/cmp_b0316.png  
      inflating: /content/data/facades/train/b/cmp_b0317.png  
      inflating: /content/data/facades/train/b/cmp_b0318.png  
      inflating: /content/data/facades/train/b/cmp_b0320.png  
      inflating: /content/data/facades/train/b/cmp_b0321.png  
      inflating: /content/data/facades/train/b/cmp_b0322.png  
      inflating: /content/data/facades/train/b/cmp_b0323.png  
      inflating: /content/data/facades/train/b/cmp_b0324.png  
      inflating: /content/data/facades/train/b/cmp_b0326.png  
      inflating: /content/data/facades/train/b/cmp_b0327.png  
      inflating: /content/data/facades/train/b/cmp_b0328.png  
      inflating: /content/data/facades/train/b/cmp_b0329.png  
      inflating: /content/data/facades/train/b/cmp_b0330.png  
      inflating: /content/data/facades/train/b/cmp_b0332.png  
      inflating: /content/data/facades/train/b/cmp_b0333.png  
      inflating: /content/data/facades/train/b/cmp_b0335.png  
      inflating: /content/data/facades/train/b/cmp_b0336.png  
      inflating: /content/data/facades/train/b/cmp_b0337.png  
      inflating: /content/data/facades/train/b/cmp_b0338.png  
      inflating: /content/data/facades/train/b/cmp_b0339.png  
      inflating: /content/data/facades/train/b/cmp_b0342.png  
      inflating: /content/data/facades/train/b/cmp_b0344.png  
      inflating: /content/data/facades/train/b/cmp_b0346.png  
      inflating: /content/data/facades/train/b/cmp_b0347.png  
      inflating: /content/data/facades/train/b/cmp_b0348.png  
      inflating: /content/data/facades/train/b/cmp_b0350.png  
      inflating: /content/data/facades/train/b/cmp_b0351.png  
      inflating: /content/data/facades/train/b/cmp_b0354.png  
      inflating: /content/data/facades/train/b/cmp_b0356.png  
      inflating: /content/data/facades/train/b/cmp_b0357.png  
      inflating: /content/data/facades/train/b/cmp_b0358.png  
      inflating: /content/data/facades/train/b/cmp_b0359.png  
      inflating: /content/data/facades/train/b/cmp_b0361.png  
      inflating: /content/data/facades/train/b/cmp_b0362.png  
      inflating: /content/data/facades/train/b/cmp_b0366.png  
      inflating: /content/data/facades/train/b/cmp_b0367.png  
      inflating: /content/data/facades/train/b/cmp_b0369.png  
      inflating: /content/data/facades/train/b/cmp_b0371.png  
      inflating: /content/data/facades/train/b/cmp_b0374.png  
      inflating: /content/data/facades/train/b/cmp_b0375.png  
      inflating: /content/data/facades/train/b/cmp_b0378.png  
      inflating: /content/data/facades/train/b/cmp_x0001.png  
      inflating: /content/data/facades/train/b/cmp_x0002.png  
      inflating: /content/data/facades/train/b/cmp_x0004.png  
      inflating: /content/data/facades/train/b/cmp_x0005.png  
      inflating: /content/data/facades/train/b/cmp_x0008.png  
      inflating: /content/data/facades/train/b/cmp_x0010.png  
      inflating: /content/data/facades/train/b/cmp_x0012.png  
      inflating: /content/data/facades/train/b/cmp_x0014.png  
      inflating: /content/data/facades/train/b/cmp_x0015.png  
      inflating: /content/data/facades/train/b/cmp_x0016.png  
      inflating: /content/data/facades/train/b/cmp_x0017.png  
      inflating: /content/data/facades/train/b/cmp_x0018.png  
      inflating: /content/data/facades/train/b/cmp_x0019.png  
      inflating: /content/data/facades/train/b/cmp_x0020.png  
      inflating: /content/data/facades/train/b/cmp_x0021.png  
      inflating: /content/data/facades/train/b/cmp_x0022.png  
      inflating: /content/data/facades/train/b/cmp_x0025.png  
      inflating: /content/data/facades/train/b/cmp_x0028.png  
      inflating: /content/data/facades/train/b/cmp_x0029.png  
      inflating: /content/data/facades/train/b/cmp_x0030.png  
      inflating: /content/data/facades/train/b/cmp_x0031.png  
      inflating: /content/data/facades/train/b/cmp_x0034.png  
      inflating: /content/data/facades/train/b/cmp_x0035.png  
      inflating: /content/data/facades/train/b/cmp_x0036.png  
      inflating: /content/data/facades/train/b/cmp_x0037.png  
      inflating: /content/data/facades/train/b/cmp_x0038.png  
      inflating: /content/data/facades/train/b/cmp_x0042.png  
      inflating: /content/data/facades/train/b/cmp_x0043.png  
      inflating: /content/data/facades/train/b/cmp_x0044.png  
      inflating: /content/data/facades/train/b/cmp_x0045.png  
      inflating: /content/data/facades/train/b/cmp_x0048.png  
      inflating: /content/data/facades/train/b/cmp_x0049.png  
      inflating: /content/data/facades/train/b/cmp_x0050.png  
      inflating: /content/data/facades/train/b/cmp_x0051.png  
      inflating: /content/data/facades/train/b/cmp_x0052.png  
      inflating: /content/data/facades/train/b/cmp_x0054.png  
      inflating: /content/data/facades/train/b/cmp_x0055.png  
      inflating: /content/data/facades/train/b/cmp_x0056.png  
      inflating: /content/data/facades/train/b/cmp_x0058.png  
      inflating: /content/data/facades/train/b/cmp_x0061.png  
      inflating: /content/data/facades/train/b/cmp_x0062.png  
      inflating: /content/data/facades/train/b/cmp_x0063.png  
      inflating: /content/data/facades/train/b/cmp_x0064.png  
      inflating: /content/data/facades/train/b/cmp_x0065.png  
      inflating: /content/data/facades/train/b/cmp_x0066.png  
      inflating: /content/data/facades/train/b/cmp_x0067.png  
      inflating: /content/data/facades/train/b/cmp_x0068.png  
      inflating: /content/data/facades/train/b/cmp_x0069.png  
      inflating: /content/data/facades/train/b/cmp_x0071.png  
      inflating: /content/data/facades/train/b/cmp_x0072.png  
      inflating: /content/data/facades/train/b/cmp_x0074.png  
      inflating: /content/data/facades/train/b/cmp_x0075.png  
      inflating: /content/data/facades/train/b/cmp_x0078.png  
      inflating: /content/data/facades/train/b/cmp_x0079.png  
      inflating: /content/data/facades/train/b/cmp_x0081.png  
      inflating: /content/data/facades/train/b/cmp_x0082.png  
      inflating: /content/data/facades/train/b/cmp_x0085.png  
      inflating: /content/data/facades/train/b/cmp_x0087.png  
      inflating: /content/data/facades/train/b/cmp_x0088.png  
      inflating: /content/data/facades/train/b/cmp_x0089.png  
      inflating: /content/data/facades/train/b/cmp_x0090.png  
      inflating: /content/data/facades/train/b/cmp_x0091.png  
      inflating: /content/data/facades/train/b/cmp_x0092.png  
      inflating: /content/data/facades/train/b/cmp_x0094.png  
      inflating: /content/data/facades/train/b/cmp_x0095.png  
      inflating: /content/data/facades/train/b/cmp_x0098.png  
      inflating: /content/data/facades/train/b/cmp_x0099.png  
      inflating: /content/data/facades/train/b/cmp_x0102.png  
      inflating: /content/data/facades/train/b/cmp_x0103.png  
      inflating: /content/data/facades/train/b/cmp_x0104.png  
      inflating: /content/data/facades/train/b/cmp_x0106.png  
      inflating: /content/data/facades/train/b/cmp_x0108.png  
      inflating: /content/data/facades/train/b/cmp_x0109.png  
      inflating: /content/data/facades/train/b/cmp_x0110.png  
      inflating: /content/data/facades/train/b/cmp_x0111.png  
      inflating: /content/data/facades/train/b/cmp_x0112.png  
      inflating: /content/data/facades/train/b/cmp_x0113.png  
      inflating: /content/data/facades/train/b/cmp_x0115.png  
      inflating: /content/data/facades/train/b/cmp_x0118.png  
      inflating: /content/data/facades/train/b/cmp_x0120.png  
      inflating: /content/data/facades/train/b/cmp_x0121.png  
      inflating: /content/data/facades/train/b/cmp_x0122.png  
      inflating: /content/data/facades/train/b/cmp_x0123.png  
      inflating: /content/data/facades/train/b/cmp_x0124.png  
      inflating: /content/data/facades/train/b/cmp_x0125.png  
      inflating: /content/data/facades/train/b/cmp_x0131.png  
      inflating: /content/data/facades/train/b/cmp_x0132.png  
      inflating: /content/data/facades/train/b/cmp_x0134.png  
      inflating: /content/data/facades/train/b/cmp_x0135.png  
      inflating: /content/data/facades/train/b/cmp_x0136.png  
      inflating: /content/data/facades/train/b/cmp_x0137.png  
      inflating: /content/data/facades/train/b/cmp_x0138.png  
      inflating: /content/data/facades/train/b/cmp_x0139.png  
      inflating: /content/data/facades/train/b/cmp_x0140.png  
      inflating: /content/data/facades/train/b/cmp_x0141.png  
      inflating: /content/data/facades/train/b/cmp_x0144.png  
      inflating: /content/data/facades/train/b/cmp_x0145.png  
      inflating: /content/data/facades/train/b/cmp_x0146.png  
      inflating: /content/data/facades/train/b/cmp_x0148.png  
      inflating: /content/data/facades/train/b/cmp_x0149.png  
      inflating: /content/data/facades/train/b/cmp_x0150.png  
      inflating: /content/data/facades/train/b/cmp_x0152.png  
      inflating: /content/data/facades/train/b/cmp_x0153.png  
      inflating: /content/data/facades/train/b/cmp_x0154.png  
      inflating: /content/data/facades/train/b/cmp_x0155.png  
      inflating: /content/data/facades/train/b/cmp_x0158.png  
      inflating: /content/data/facades/train/b/cmp_x0162.png  
      inflating: /content/data/facades/train/b/cmp_x0165.png  
      inflating: /content/data/facades/train/b/cmp_x0166.png  
      inflating: /content/data/facades/train/b/cmp_x0167.png  
      inflating: /content/data/facades/train/b/cmp_x0168.png  
      inflating: /content/data/facades/train/b/cmp_x0170.png  
      inflating: /content/data/facades/train/b/cmp_x0171.png  
      inflating: /content/data/facades/train/b/cmp_x0172.png  
      inflating: /content/data/facades/train/b/cmp_x0173.png  
      inflating: /content/data/facades/train/b/cmp_x0174.png  
      inflating: /content/data/facades/train/b/cmp_x0175.png  
      inflating: /content/data/facades/train/b/cmp_x0176.png  
      inflating: /content/data/facades/train/b/cmp_x0178.png  
      inflating: /content/data/facades/train/b/cmp_x0179.png  
      inflating: /content/data/facades/train/b/cmp_x0180.png  
      inflating: /content/data/facades/train/b/cmp_x0181.png  
      inflating: /content/data/facades/train/b/cmp_x0182.png  
      inflating: /content/data/facades/train/b/cmp_x0185.png  
      inflating: /content/data/facades/train/b/cmp_x0187.png  
      inflating: /content/data/facades/train/b/cmp_x0189.png  
      inflating: /content/data/facades/train/b/cmp_x0190.png  
      inflating: /content/data/facades/train/b/cmp_x0191.png  
      inflating: /content/data/facades/train/b/cmp_x0192.png  
      inflating: /content/data/facades/train/b/cmp_x0193.png  
      inflating: /content/data/facades/train/b/cmp_x0194.png  
      inflating: /content/data/facades/train/b/cmp_x0195.png  
      inflating: /content/data/facades/train/b/cmp_x0196.png  
      inflating: /content/data/facades/train/b/cmp_x0199.png  
      inflating: /content/data/facades/train/b/cmp_x0200.png  
      inflating: /content/data/facades/train/b/cmp_x0201.png  
      inflating: /content/data/facades/train/b/cmp_x0202.png  
      inflating: /content/data/facades/train/b/cmp_x0203.png  
      inflating: /content/data/facades/train/b/cmp_x0204.png  
      inflating: /content/data/facades/train/b/cmp_x0206.png  
      inflating: /content/data/facades/train/b/cmp_x0208.png  
      inflating: /content/data/facades/train/b/cmp_x0209.png  
      inflating: /content/data/facades/train/b/cmp_x0211.png  
      inflating: /content/data/facades/train/b/cmp_x0212.png  
      inflating: /content/data/facades/train/b/cmp_x0215.png  
      inflating: /content/data/facades/train/b/cmp_x0218.png  
      inflating: /content/data/facades/train/b/cmp_x0219.png  
      inflating: /content/data/facades/train/b/cmp_x0222.png  
      inflating: /content/data/facades/train/b/cmp_x0225.png  
      inflating: /content/data/facades/train/b/cmp_x0227.png  
      inflating: /content/data/facades/train/b/cmp_x0228.png  
    

# < PIx2PIx >  

pix2pix에 대해 간략하게 말하자면, 왼쪽사진이 입력으로 들어가는 조건 이미지 인데, 이러한 이미지가 입력으로 들어왔을 때 오른쪽과 같이 사진으로 찍은거 같은 이미지가 출력될 수 있도록 하는 것.

이미지는 위에서 본것처럼 입력과 출력형태의 이미지가 두 개가 붙어있는 형태이지만, 학습을 진행할때는 따로따로 봐야되므로 이미지를 네트워크에 넣기 위해 custom dataset을 정의해준다.  

> <Dataset & Dataloader>
> Dataset 은 샘플과 정답(label)을 저장하고, DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있는 객체(iterable)로 감쌉니다.  

> - Dataset : 전체 dataset을 구성하는 단계로 dataloader를 통해 data를 받아오는 역할을 합니다. 

>> <Dataset class에서 반드시 정의해야 하는 method 3가지>
* init(self) : 필요한 변수 선언
* get_item(self,index) : 만든 리스트의 index에 해당하는 샘플을 데이터셋에서 불러오고 전처리하여 tensor(배열, 행렬) 자료형으로 바꿔서 리턴
* len(self) : 학습 데이터 개수 리턴  

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




    
![png](output_11_1.png)
    


## Dataloader  

이제 데이터셋을 불러와 앞에서 만든 custom dataset에 넣은 것을 train_ds에 저장하고 train_ds를 사용해 dataloder를 만들어 train_dl에 저장한다.


```python
# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
```

# < 모델 구축하기 >

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWMAAAD/CAYAAAAkEbdtAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAG4dSURBVHhe7d13sCVHlSf+X8ROrPkFv5n9Y3djZ5lhJmZjYBdmYAYjhIQkBAzy3iCDhJDBySDkvVoOOWSxQjgh7733EvIt1/Let7wXoAFm8peffO901yvu7Xef6373vsyIb1RVVlbmOZknv3UqKyvr/6mhhhpqqGGahD/84Q+fyZhVUVFRMZn4t3/7t3GjU36DiD/+8Y97/Pu///t7CxlTPB+kioomsl1U9Ck6tWfF9ERur3fzdrFCximlWRk11DAitI2mon9QQ/+E3F6VjGuooYbBDM0b08LEeEK+rpJxDTXUMLWhSVRjxUwJWddKxjXUUEMNizpUMq6hhhpqmAahknENNdRQwzQIlYxrmHDIBlSmUf3xj38cMaUqQpzvdK5baF8jNK+P85FXc7+GGhZ1CBsdC7INVzKuYeIBGf7hD3+YZ1jtMGxs8xBpIn0zLkLzHMgf4Ue6ZvpO1481NPMdVNQwfUNun0rGNYw/BFEiyaZnHBCQQPOcfcQa5yKuSRbignzjfMCxENd2QjOvXkNcYzuoqGH6htw+lYxrGF/Qud999930m9/8Jv32t79N//qv/5p+97vfpXfeeafE/f73vy9pbJ2XNtI4DnKwbZIoSPf666+nN954o6SXR6SJfBGzPJVnK8/Yl67XEGXGflOW2O9XhG6dUMP0CrlNKhnXMLYQnRz5Pfjgg+n8889PRx99dNpzzz3Td7/73XTKKaek2bNnF0KW9qmnnkoXXnhhOvDAA9N+++2Xfvazn6Vbb721EG6TGMLrFf/yyy+n6667Lv30pz9N++yzT9pjjz3SHXfcMeIaZK2cH/zgByXfI488ssjy0ksvlXRC5N1LiHyDyNqI8/2MmRY61cF0RbaxSsY1jC2E8fBYn3322XTTTTelWbNmpcUXXzxts8026ayzzkqPP/54IWsEi1ivv/76tO+++6bddtstHXfccenRRx/9EzIO0uP58ogR/Yknnpi+8Y1vpE9+8pPpgAMOSHfeeWdJawjj7bffLmlOOumktMsuu6Tvfe976eabb05vvfXWCE8aohz7QrPcF198Mc2dO7eU6XxzOKR57SBgpoVOdTBdkW2tknENYw+MB2EhRUMDiPBv/uZv0s4771wIE4kZLnDePsI7/fTTCxHfcsst8663bYZho5w3FCEvJLzUUkulFVZYoXi/yD2I3vaxxx4rHvQ111xTjoWQTV5NYo0y41ggjxvK008/Pe+6uLYtXw39FbTfosB4Qr6uknENYw+IKggLcR577LHpgx/8YBkuuPfee+cRJe9XGl4n7/iqq65KDz/88Lw8Ohmv4yBNac8999zibX/hC19Iq6yySjrzzDPTm2++WdKS4dVXX02XX355uu++++blJW/oFKLcCPIzvMHLd44+5JZ3EHlbxhpqmOyQbayScQ1jDwgqyBhx/eQnP0kf+MAH0t57753uueeeQmLNF2+vvfZauvrqq9Oll16aHnjggXl5BNpkF8dz5sxJZ599dvFaDz/88LTWWmulNdZYI911111lTFrZvG5j0tJGXkAGaQyJIGrEbjyZTM4hcV7xt771rbT//vuXcWzlkJXs8ugkW6fQKU0v1wndrh1vuTX0Z8htWcm4hrEHJNUk42OOOSa9//3vT3vttVchxTYZI74rr7yykCZiFKSRR8CxtM2AdJGxl3W333572n333dM//uM/psMOO6yMFwcZ854NacgDQcnvySefLJ64G4CXgfbPO++89NBDDxXCNa5teGP55ZdPm2++efG+b7jhhnKdoRehV7LrlG6i1/Zyfa9l1DD9Q27LSsY1jD0ggSA+hNv0jJtkjCwRLPJDhkHGrkOYzhtbDkSeEeSFaJ955pmS30UXXVTGjpdZZpkya8PLOkSNZJG1PAUeMHLdaaed0qmnnppuu+22dPHFF6ddd901/fKXvywyPPLII8XbXnrppdMGG2yQjj/++HTJJZcUso5hkKYsnYLzgXZonhvtfDs0z412vobBCLktKxnXMPaAYBGngESRsTHjIGPnxTeHKZAxQrz//vvLdU1CaaIZ5MUz5q0agzbrAbl+6EMfSltttVXxZJE0wka4yF047bTTygwLnrSxYGl4wmZnbLbZZumcc84p3q+hi6985StlNojzZmjE8EcneWqoYapCtrVKxjWMPXQjY3ON77777q5kHJ4xcvSi79prry0v34wnI2qEinAFRCgvxBmesTx5tNtvv31ad911S3mOpTFMoTzBi8QNN9ywDJsgf0Qrb2Qs/uSTTy7erzFi5GzanTLoFEMmQiXj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)

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

![image.png](data:image/png;base64,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)

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
<font color='red'> stage를 거칠수록 채널수는 늘어나고 이미지의 높이와 너비는 반씩 줄어들게되어 stage4에서는 512 x 16 x 16형태가 되고 이를 self.patch를 통해 1 x 16 x 16으로(압축)만들어주게 되고 마지막으로 sigmoid를 취해 각 픽셀값을 sigmoid형태로 해줍니다.


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


    
![png](output_40_0.png)
    



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


    
![png](output_45_0.png)
    

