---
title:  "Stereo Camera Calibration"
categories:
  - computervision
---

# Stereo Camera Calibration  

이번에는 Stereo Camera Calibration에 대해 알아보자. 그 전에 저번에 공부했던 Camera Calibration에 대해 잠깐 짚고 넘어가자.  

## < Camera Calibration >  

Camera Calibration이란 카메라가 가지고 있는 Parameter를 구하는 것을 의미한다.  
실제로 블로그의 Camera Calibration 편을 보면 내부 요인의 파라미터 값을 구하는 과정을 카메라 캘리브레이션이라 부른다고 하였지만 
실제로는 내부 파라미터와 외부 파라미터가 다 구해진다.  

## < 카메라 내부 파라미터(intrinsic parameters) >  

intrinsic camera matrix A는 아래와 같은 행렬이며 그 구성은 다음과 같다.  

![image](https://user-images.githubusercontent.com/93988405/235559468-8c5ac564-d9b5-4132-8dc3-47eafadc625c.png){:class="align-center"}  

> - 초점거리(focal length): fx, fy  
> - 주점(principal point): cx, cy  
> - 비대칭계수(skew coefficient): skew_c = tanα  

## < 카메라 외부 파라미터(extrinsic parameters) >  

extrinsic camera calibration은 아래 그림과 같이 크기 및 형태를 미리 알고 있는 외부의 물체(마크)에 대한 영상을 분석하여 이 영상을 획득할 당시의 카메라의 3차원 위치(지면으로부터 높이 등) 및 3D 자세 정보(팬, 틸트 등)를 추출하는 것 이다.  

카메라 외부 파라미터는 카메라 좌표계와 월드 좌표계 사이의 변환 관계를 설명하는 파라미터로서, 두 좌표계 사이의 회전(rotation) 및 평행이동(translation) 변환으로 표현된다.  

카메라 외부 파라미터를 구하기 위해서는 먼저 캘리브레이션 툴 등을 이용하여 카메라 고유의 내부 파라미터들을 구한 다음 미리 알고 있는 또는 샘플로 뽑은 3D월드좌표–2D영상좌표 매칭 쌍들을 이용하여  

![image](https://user-images.githubusercontent.com/93988405/235559631-7fd4bbad-4750-4ec5-8aba-3fb7257659d9.png){:class="align-center"}  

위의 식에서 변환행렬을 구하면 된다.  

## < Stereo Camera Calibration >  

![image](https://user-images.githubusercontent.com/93988405/235559685-3b9f713a-53f7-4a71-95f6-40c781006a5e.png){:class="align-center"}  

![image](https://user-images.githubusercontent.com/93988405/235559698-7a4c0697-3b36-4c95-ab2d-674be79c6a94.png){:class="align-center"}  

2대 이상의 카메라의 위치관계 즉, 각 카메라의 상대적 위치를 알아내는 것을 Stereo Camera Calibration이라 한다. 진행과정은 다음과 같다.  

1. 먼저 각 두 대의 카메라가 한 체커보드를 보고 있다 가정하면, 각 카메라가 보고있는 체커보드의 사진을 여러장찍는다.  
2. Zhang’s method를 사용해 각각의 카메라의 Self Calibration을 진행한다.  
3. 이제 각 카메라간의 관계를 알아내는데 1번에서 찍은 사진쌍들중에 아무 사진이나 갖고 진행해도 무방하다.  
4. 각각의 카메라의 내부변수와 위치추정값을 최적화해가며 값을 찾는다.  

최적화 과정은 다음과 같다.  

![image](https://user-images.githubusercontent.com/93988405/235559773-7c8cab0e-9e9f-4dbd-9001-1d931c2eb24f.png){:class="align-center"}  

먼저 왼쪽 카메라 좌표 XL을 image plane에 투영한 것을 uL이라하고 오른쪽 카메라 좌표 XR을 image plane에 투영한 것을 uR이라 하자.  
우리는 위에서 구한 Rotation과 Translation 정보를 담고 있는 T matirx를 통해 XL을 XR로 추정할 수 있고 이 추정한 XR로 uR값을 구할 수 있다.  
이렇게 추정한 uR, uL값을 실제 uR, uL값들과 비교해가며 값을 최적화 해나간다.  

실제로 결과를 확인해보면 다음과 같다.  
좌표가 있는 real image point는 파란색으로 표시하였고 구한 stereo calibration을 통해 얻은 projected image point는 빨간색으로 표시하였다.  

![image](https://user-images.githubusercontent.com/93988405/235559894-4263f616-2d20-4071-8090-14cf938a9d90.png){:class="align-center"}  

이를 확대하면 아래와 같다.  

![image](https://user-images.githubusercontent.com/93988405/235559942-3d191276-0ef7-41de-9129-f940e568b2cc.png){:class="align-center"}  
