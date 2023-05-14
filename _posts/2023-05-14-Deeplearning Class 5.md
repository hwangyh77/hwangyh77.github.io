---
title:  "Deeplearning class 5"
categories:
  - deeplearning
---

# Convolutional Neural Network  

오늘은 딥러닝의 기본 모델 구조들 중 CNN(Convolutional Neural Network)에 대해 알아보자.  

## < CNN >  

CNN은 Convolutional Neural Networks의 약자로 딥러닝에서 주로 이미지나 영상 데이터를 처리할 때 쓰이며 Convolution이라는 전처리 작업이 들어가는 Neural Network 모델이다.  
CNN은 이미지를 날것(raw input) 그대로 받음으로써 공간적/지역적 정보를 유지한 채 특성(feature)들의 계층을 빌드업한다.  
<font color='red'>CNN의 중요 포인트는 이미지 전체보다는 부분을 보는 것, 그리고 이미지의 한 픽셀과 주변 픽셀들의 연관성을 살리는 것이다.</font>  

![image](https://github.com/hwangyh77/hwangyh77.github.io/assets/93988405/13cd70f6-7f13-40c5-8db8-ab6d47d9a8b5){:class="align-center"}  

위와 같은 모델 구조를 갖고 있다. 위에서 말했듯이 CNN은 데이터의 특징을 추출하여 특징들의 패턴을 파악하는 구조이다.  
이 CNN 알고리즘은 Convolution Layer와 Pooling Layer를 복합적으로 구성하여 알고리즘을 만든다.  

## < Convolution Layer >  

![image](https://github.com/hwangyh77/hwangyh77.github.io/assets/93988405/e5db8188-5966-405e-afe5-b8b109acea35){:class="align-center"}  

기본적으로 위와 같이 Convolution이 진행되며 아래와 같이 만들어진다. 
하나의 합성곱 계층에는 입력되는 이미지의 채널 개수만큼 필터가 존재하며, 각 채널에 할당된 필터를 적용함으로써 합성곱 계층의 출력 이미지가 생성된다.  

![image](https://github.com/hwangyh77/hwangyh77.github.io/assets/93988405/c8c028b5-f16a-4854-9aa3-08aa3d0c63fe){:class="align-center"}  

> Convolution 했을 때 output size를 구하는 식은 다음과 같다.  
> 
> ![image](https://github.com/hwangyh77/hwangyh77.github.io/assets/93988405/a0f150a7-f58b-4b91-9767-fe31461507c1){:class="align-center"}  
>  
> - Stride : 이미지에 대해 필터를 적용할 때 필터의 이동량  
> - Padding : convolution 진행 시 출력 이미지의 크기는 입력 이미지의 크기보다 작아지므로 convolution layer를 거치면 거칠수록 점점 작아지게 된다.
> 이러한 문제점을 해결하기 위해 입력 이미지의 가장자리에 특정 값( 0 )으로 설정된 피겔들을 추가하는 것을 (zero)padding이라 한다.  

## < Pooling Layer >  

이미지의 크기를 계속 유지한 채 Fully Connected layer로 가게되면 연산량이 기하급수적으로 늘어날 것이다.  
이에 적당히 크기도 줄이고, 특정 feature를 강조하는 역할을 수행하는 것이 Pooling이다.  
> - Pooling = Subsampling(해당 image data를 작은 size의 image로 줄이는과정)  
> 왜냐?  
> 앞선 layer 들을 거치고 나온 output feature map의 모든 데이터가 필요하지 않기 때문 -> 추론하는데 적당량의 data만 있어도 되기 때문!!  

Pooling의 종류에는 Max, Average, Min pooling들이 있다. 그중 Max pooling의 작동방식을 보면 다음과 같다.  

![image](https://github.com/hwangyh77/hwangyh77.github.io/assets/93988405/f4617418-1b22-454c-93aa-f049e180c03a){:class="align-center"}  

CNN에서는 주로 Max pooling을 사용하는데, 이는 뉴런이 가장 큰 신호에 반응하는 것과 유사하다 한다. 이렇게 하면 노이즈도 감소하고 속도가 빨라진다.  

## < Flattening Layer >  

![image](https://github.com/hwangyh77/hwangyh77.github.io/assets/93988405/dc586fa3-9dab-4430-9131-59a5ee13c837){:class="align-center"}  

Flattening Layer의 목적은 Convolutional Layer, Pooling Layer를 feature를 추출한 다음, 
추출된 특성을 Output Layer에 연결(fully connected layer)하여 어떤 이미지인지 분류하기 위해서이다.  

위에서 정리한 개념들을 갈무리하자면  

CNN은 기본골격으로 <mark style='background-color: #dcffe4'>Input -> Padding -> Convolution Layer -> Pooling -> Convolution Layer -> Pooling -> ... -> Flattening -> Output</mark>의 순서로 진행된다.
