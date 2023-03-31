---
title:  "Camera Calibration"
categories:
  - computervision
---

# Camera Calibration  

## < Camera Calibration 이란? >  

실제 눈으로 보는 세상은 3차원. But 이것을 카메라로 찍으면 2차원이다.  

이 때, 3차원의 점들이 이미지 상에서 어디에 맺히는지는 기하학적으로 생각하면 
영상을 찍을 당시의 카메라의 위치 및 방향에 의해 결정되는데 실제 이미지는 사용된 렌즈, 
렌즈와 이미지 센서와의 거리, 렌즈와 이미지 센서가 이루는 각 등 카메라 내부의 기구적인 부분에 의해서
크게 영향을 받는다.  

3차원 점들이 영상에 투영된 위치를 구하거나 역으로 영상좌표로부터 3차원 공간좌표를 복원할 때 이러한 내부 요인을 제거해야만 정확한 계산이 가능해진다. 
이러한 <font color='red'>내부 요인의 파라미터 값을 구하는 과정을 카메라 캘리브레이션이라 부른다.</font>  

![image](https://user-images.githubusercontent.com/93988405/219863799-ba76e160-9be4-40cc-9663-b0ec8e8bcb3f.png){:class="align-center"}  

여기서, (X,Y,Z)는 월드 좌표계(world coordinate system) 상의 3D 점의 좌표, [ R | t ]는 월드 좌표계를 카메라 좌표계로 변환시키기 위한 회전/이동변환 행렬이며 A는 intrinsic camera matrix이다.
<mark style='background-color: #dcffe4'>A와 [R|t]를 합쳐서 camera matrix 또는 projection matrix라 부른다.</mark>  

![image](https://user-images.githubusercontent.com/93988405/219863862-10dcedf3-ea7f-4543-bd30-ad8d2e6d946d.png){:class="align-center"}  

카메라 캘리브레이션(camera calibration)은 위와 같은 3D 공간좌표와 2D 영상좌표 사이의 변환관계 또는 이 변환관계를 설명하는 파라미터를 찾는 과정이다.  

## < 카메라 내부 파라미터(intrinsic parameters) >  

intrinsic camera matrix A는 아래와 같은 행렬이며 그 구성은 다음과 같다.  

![image](https://user-images.githubusercontent.com/93988405/219863933-c9ee730f-9f9c-4cc7-bbf7-a72cd914abf0.png){:class="align-center"}  

- 초점거리(focal length): fx, fy  
- 주점(principal point): cx, cy  
- 비대칭계수(skew coefficient): skew_c = tanα  

#### 1. 초점 거리(focal length)  

![image](https://user-images.githubusercontent.com/93988405/219863997-d9ec2a84-b85e-4913-95f6-bc3f362cfb7a.png){:class="align-center"}  

- 렌즈중심과 이미지센서(CCD, CMOS 등)와의 거리를 말합니다.  
<font color='red'>카메라 모델에서 말하는 초점거리(f)는 픽셀(pixel) 단위로 표현됩니다.</font> 즉, f의 단위로 픽셀이라는 의미  

초점 거리를 fx, fy로 구분하여 표현하는 이유는 이미지 센서의 물리적인 셀 간격이 가로 방향과 세로 방향이 서로 다를 수 있음을 모델링하기 위함이다.
<font color='red'>(현대의 일반적인 카메라는 가로방향 셀 간격과 세로방향 셀 간격의 차이가 없기 때문에 f = fx = fy라 놓아도 무방)</font>  

> **Normalized image plane**
> ![image](https://user-images.githubusercontent.com/93988405/219864046-385fb68f-a547-492f-9028-87a6c588e770.png){:class="align-center"}  
>실제는 존재하지 않는 가상의(상상의) 이미지 평면으로 초점으로부터 거리가 1(unit distance)인 평면을 normalized image plane이라고 부른다.
>카메라 좌표계 상의 한 점 (Xc, Yc, Zc)를 영상좌표계로 변환할 때 먼저 Xc, Yc를 Zc(카메라 초점에서의 거리)로 나누는 것은 이 normalized image plane 상의 좌표로 변환하는 것이며, 
>여기에 다시 초점거리 f를 곱하면 우리가 원하는 이미지 평면에서의 영상좌표(pixel)가 나온다.  

#### 2. 주점(principal point)  

주점 cx, cy는 카메라 렌즈의 중심 즉, 핀홀에서 이미지 센서에 내린 수선의 발의 영상좌표(단위는 픽셀)이다.  

#### 3. 비대칭 계수(skew coefficient)  

![image](https://user-images.githubusercontent.com/93988405/219864112-fad77c55-1918-452f-9131-11ff4b94b20f.png){:class="align-center"}  

요즘 카메라들은 이러한 skew 에러가 거의 없기 때문에 카메라 모델에서 보통 비대칭 계수까지는 고려하지 않는다. (즉, skew_c = 0)

## < 카메라 외부 파라미터(extrinsic parameters) >  

extrinsic camera calibration은 아래 그림과 같이 크기 및 형태를 미리 알고 있는 외부의 물체(마크)에 대한 영상을 분석하여 
이 영상을 획득할 당시의 카메라의 3차원 위치(지면으로부터 높이 등) 및 3D 자세 정보(팬, 틸트 등)를 추출하는 것이다.  

카메라 외부 파라미터는 카메라 좌표계와 월드 좌표계 사이의 변환 관계를 설명하는 파라미터로서, 
두 좌표계 사이의 회전(rotation) 및 평행이동(translation) 변환으로 표현된다.  

카메라 외부 파라미터를 구하기 위해서는 먼저 캘리브레이션 툴 등을 이용하여 
카메라 고유의 내부 파라미터들을 구한 다음 미리 알고 있는 또는 샘플로 뽑은 3D월드좌표–2D영상좌표 매칭 쌍들을 이용하여  
![image](https://user-images.githubusercontent.com/93988405/219864180-0dbe6446-0727-434d-93de-be0ac1810069.png){:class="align-center"}  
위의 식에서 변환행렬을 구하면 된다.  

## < 3D 공간좌표와 2D 영상좌표 >  

![image](https://user-images.githubusercontent.com/93988405/219864207-d344960f-4ab0-451f-ab18-3cbde7bd78ff.png){:class="align-center"}  

카메라를 중심으로 3차원 좌표계(카메라의 광학축 방향이 Z축, 오른쪽이 X축, 아래쪽이 Y축)를 설정했을 때, 외부의 한점 P(X,Y,Z)에 대한 영상좌표를 구해보자.  

P(X,Y,Z)는 카메라로부터 (수직방향으로) Z만큼 떨어진 거리에 있는 점이므로 카메라로부터의 거리가 1인 가상의 정규 이미지 평면에서의 좌표는 
삼각형의 닯음비를 이용하면 P'(X/Z,Y/Z,1)이 된다. But 실제 카메라 영상은 초점거리 f만큼 떨어진 이미지 평면에 투사되므로 
P에 대응되는 이미지 좌표는 P''(f*X/Z, f*Y/Z, f)가 되고 이를 픽셀좌표계로 변환하면 우리가 원하는 최종 영상좌표를 얻을 수 있다.
P''를 픽셀좌표계(이미지의 좌상단이 원점)에 맞추어 주면 영상중심을 (cx, cy)라 했을 때 x = fx*X/Z + cx, y = fy*Y/Z + cy가 최종 영상좌표가 된다.  

역으로 2D 영상좌표 p(x, y)에 대응하는 3D 공간좌표를 구하고자 할 때에는 
먼저 (x-cx, y-cy)로 영상좌표 원점을 영상중심으로 옮긴 후 Z거리가 1인 정규이미지 평면에서의 좌표 ((x-cx)/fx, (y-cy)/fy)로 바꾼다. 
다음으로 실제 물체와의 거리 Z를 반영해 주면 ((x-cx)/fx*Z, (y-cy)/fy*Z, Z)가 구하고자 하는 3차원 공간좌표가 된다.  

## < 카메라의 3D 위치 파악 >  

3D 월드좌표를 3D 카메라좌표로 변환시키는 회전변환 R과 평행이동 T를 계산했다고 가정하자.  

차원 공간상의 한점 P에 대한 월드좌표를 Pw = [xw, yw, zw]T, 카메라 좌표계에서 봤을 때의 좌표를 Pc = [xc, yc, zc]T, solvePnP 함수가 반환하는 회전변환 행렬을 R, 평행이동 벡터를 T라고 했을 때 다음과 같은 변환 관계식이 성립한다.  

![image](https://user-images.githubusercontent.com/93988405/219864322-1adf50dd-eaed-4019-8586-2a7b3783f7ea.png){:class="align-center"}  

이 때, 카메라의 3D 위치(월드좌표)는 카메라 좌표계의 원점에 대응하는 월드좌표이므로 다음과 같이 계산된다.  
![image](https://user-images.githubusercontent.com/93988405/219864345-674b5b18-7ae9-4471-90dd-2a845c97eb6a.png){:class="align-center"}  

## < References >  

https://darkpgmr.tistory.com/32
