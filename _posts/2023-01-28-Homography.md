---
layout: post
title:  "Image Homography 공부"
categories:
  - computervision
---

# Image Homography    

![image](https://user-images.githubusercontent.com/93988405/215250739-039db9a0-26b6-40ce-91fd-7bf679e7d6be.png){:class="align-center"}  
 
 값을 변화시키는 함수를 h 로, 이미지를 F 라고 했을 때 Filtering 은 이미지 픽셀 값들에 h 를 적용하는 것이다. 예를 들어, h(x)=2x 라고 한다면 2 배가 밝아진 이미지가 되는 것이다. 이를 이미지 함수의 Range 를 변경한다고 표현. 그리고 Warping 은 좌표에 h 를 적용하는 것이기 때문에 똑같이 h(x)=2x 라면 반이 줄은 이미지가 된다. 이를 이미지 함수의 Domain 을 변경한다고 표현.  
 
## < Image Warping(2D Transformations) >  
 
 ![image](https://user-images.githubusercontent.com/93988405/215250801-e93a7bd8-9989-4bde-9863-14c357f41ef5.png){:class="align-center"}  

위의 사진과 같이 다양한 종류의 2D Transformation 이 존재한다.  

## < Image homography >  

![image](https://user-images.githubusercontent.com/93988405/215250848-5ccb58b9-b643-4859-a872-8eaddbc8b673.png){:class="align-center"}  

맨 위의 이미지들을 정말 순서에 맞게 그대로 가져다가 이어 붙인다면 어떠한 결과가 나올까? 중간의 계단 부분만 붙여본 결과 왼쪽 아래의 그림과 같이 중앙 오른쪽을 보면 맞물려 있지 않고 엇갈려 있어 부자연스러운 느낌이 든다. 우리가 붙인 두 사진의 각도가 다르기 때문인데, 실제로 우리가 이미지를 찍었을 때 눈으로 봤던 것과 같은 넓은 전경을 가지기 위해서는 각도를 맞춰주어야 한다. 이때 사용되는 것이 바로 Image Homography 이다. homography를 사용함으로써 오른쪽 아래와 같이 각도를 맞춰줄 수 있다.  

![image](https://user-images.githubusercontent.com/93988405/215250897-fc433b44-2e15-4703-b3dd-ea3227678446.png){:class="align-center"}  

Homography 의 기본적인 아이디어는 다음 사진과 같다. PP3 라는 사진의 평면을 서로 다른 방향에서 찍었다고 하자. 서로 다른 지점에서 보고 있는 방향을 나타내는 벡터를 법선벡터로 가지는 평면이 있을 것이다. 간단히 상상해보면 향하고 있는 방향에 평행한 평면 PP1, PP2 가 있다. 그러면 해당 지점에서 찍은 사진은 평면 원본 사진 PP3 를 각 평면 PP1, PP2 에 Project 한 평면일 것이다.
<font color='red'>즉 Homography 는 바로 Projective Transformation 을 의미하는 것이다.</font>  

쉽게 말하자면 한 평면을 다른 평면에 투영(projection)시켰을 때 투영된 대응점들 사이에는 일정한 변환관계가 성립하는데 이 변환관계를 호모그래피라 부른다.
<font color='red'>호모그래피(homography)는 3×3 행렬로 표현되며 대응점들의 동차좌표(homogeneous coordinate) 표현에 대해 성립하는 변환관계이다.</font>  

변환 즉 Transformation에는 여러 종류가 있다. 그 중 몇가지를 살펴보면  

### 1. 강체 변환(Rigid-Body Transformation)  

![image](https://user-images.githubusercontent.com/93988405/215250955-bfef1fcc-16dc-4cb8-bc84-8defd7267e3a.png){:class="align-center"}  

체의 평행 이동(Translation), 회전(Rotation)등 물체의 위치와 방향만 바꾸고 형태가 변하지 않는 변환으로 변환에 대한 매핑 함수는 위의 식과 같다.  

회전 변환을 먼저 보면 회전 변환은 물체 자체가 회전하는 것이 아니라 물체가 속한 좌표계의 원점(0,0)을 기준으로 좌표계가 회전하는 것이다. 따라서 두 좌표의 원점간 거리가 다르면 두 좌표간의 관계를 회전변환만으로는 표현할 수 없다. 이를 해결하기 위해서는 두 좌표의 원점간 거리(Scale)을 고려한 매핑 함수를 사용하고 scale 변화 값을 1로 계산하여 두 좌표간 회전각을 구할 수 있다.  

![image](https://user-images.githubusercontent.com/93988405/215251033-48808537-9d08-4776-b595-dbaa57928b1e.png){:class="align-center"}  

왼쪽 위의 식을 정리하면 오른쪽 위의 식과 같게 되고
따라서 앞의 평행이동과 회전변환을 모두 고려한 매핑함수는 다음과 같이 표현 가능하다. 이 행렬식을 전개하면 ![image](https://user-images.githubusercontent.com/93988405/215251087-d555c6b1-0ca8-4692-a3c5-73790336d2c3.png)
 이며
 이를 행렬식으로 다시 나타내면 위의 식들의 아래쪽과 같다.
 
 ### 2. 투영 변환(Projective Transformation)  
 
 ![image](https://user-images.githubusercontent.com/93988405/215251162-320ee744-efbd-402d-8a5b-eb4ad94f0135.png){:class="align-center"}
 
 
Homography 라고도 하며 , 3D 공간에서 2D 공간으로 투영하거나, 서로 다른 두 평면 간의 매핑 관계를 모델링하기 위한 변환이다. 우리의 두 눈은 물체의 서로 다른 평면을 보고 있으며 이로 인해 평행성이 깨지고 물체에 원근감이 생기게 된다. 우리에게서 멀어질수록 물체가 한 점으로 모이게 되는 것처럼 보이는 것이 바로 이 투영 변환과 같은 원리이다.  

Homography는 8자유도를 가지며, 4쌍의 좌표를 통해 변환 관계를 정의한다. <mark style='background-color: #dcffe4'>(9 자유도가 아닌 이유는 동차 좌표계(Homogeneous Coordinate)에서 정의되어 9개의 요소 중 scale factor 값을 무시해도 되기 때문.)</mark>  

> - 동차좌표계
> 3차원의 한 좌표를 특정 방향으로 2차원 평면으로 투영할 때 동일한 방향으로 무한대 거리에 있는 평면들을 표현가능한 좌표계.

호모그래피 함수는 아래와 같이 표현된다.  
![image](https://user-images.githubusercontent.com/93988405/215251274-f0b40e65-1258-41b5-b5ac-57c5647ac75c.png){:class="align-center"}  

이 식을 DLT(Direct Linear Transformation) 방법을 이용해 다음과 같이 변형가능하다.  

![image](https://user-images.githubusercontent.com/93988405/215251358-acc3d6a8-6e37-4419-a34a-07ef3efee703.png){:class="align-center"}  

따라서 Homography 함수는 좌표 성분 A에 대해 의 가장 작은 eigenvalue를 가지는 eigenvector가 된다.  


### Homography에대해 좀더 자세히 보자  


![image](https://user-images.githubusercontent.com/93988405/215251645-89d501b2-5b17-493b-8c33-50ed0290ff8d.png){: width="80%" height="80%"}{:class="align-center"}  

![image](https://user-images.githubusercontent.com/93988405/215251523-95ac2af3-5cf9-44a1-896c-1f08a1f43562.png){: width="80%" height="80%"}{:class="align-center"}   

![image](https://user-images.githubusercontent.com/93988405/215251557-782b2732-89b4-4780-91b4-905b5a0478a9.png){: width="80%" height="80%"}{:class="align-center"}  
