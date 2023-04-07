---
title:  "Deeplearning class 2"
categories:
  - deeplearning
---

# Perceptorn & Multilayer Perceptron  

오늘은 딥러닝에서 퍼셉트론과 다층 퍼셉트론에 대해 공부해보자!!

## < Perceptron >  

퍼셉트론(Perceptron)은 인공 신경망(Aritificial Neural Network, ANN)의 구성 요소(unit)로서 
다수의 값을 입력받아 하나의 값으로 출력하는 이진 분류(Binary Classification) 모델을 학습하기 위한
지도학습(Supervised Learning) 기반의 알고리즘이다.  

![image](https://user-images.githubusercontent.com/93988405/230609821-8231a6cc-ac09-4bdd-aa58-d3d3d88c0358.png){:class="align-center"}  

퍼셉트론은 위와 같이 도식화 할 수 있다.
 뉴런이 다른 뉴런으로부터 신호를 입력받듯 다수의 값 x를 입력받고, 입력된 값마다 가중치(weight)를 곱한다.
 이렇게 입력받은 값을 모두 합산하는데, 합산된 결과값을 가중합 이라 부른다.  

## < Multi-layer Perceptron XOR problem >  

![image](https://user-images.githubusercontent.com/93988405/230609958-4a8d3807-00b5-4747-a177-19dec94b5fa5.png){:class="align-center"}  

위와같은 도식의 XOR 표를 작성해보면  

![image](https://user-images.githubusercontent.com/93988405/230610008-a1fd8f0a-2f08-4307-b0e4-56e5a18cada2.png){:class="align-center"}  

위와같이 XOR 다층 문제가 해결된다. 이외에도 다른 가중치를 적용하여 n1과 n2에 적절한 Threshold를 설정하는 것으로도 해결된다.  

## < Multilayer Perceptron >  

![image](https://user-images.githubusercontent.com/93988405/230610076-5e88a91b-bf36-43d2-936a-dcbd43da81ad.png){:class="align-center"}  


다층 퍼셉트론과 단층 퍼셉트론의 차이는 단층 퍼셉트론은 입력층과 출력층만 존재하지만, 다층 퍼셉트론은 중간에 층을 더 추가하였다는 점이다.
이러한 입력층과 출력층 사이에 존재하는 층을 은닉층(hidden layer)이라고 한다.  

또한 위의 그림과 같이 모든 input unit 들이 output unit 들에 연결된 구조를 fully connected layer라고 부른다.
<font color='red'>이렇게 fully connected layer로 구성된 multilayer network를 multilayer perceptron이라 부른다.</font>  

## < Activation Functions >  

딥러닝 네트워크에서 노드에 입력된 값들을 비선형 함수에 통과시킨 후 다음 레이어로 전달하는데, 이 때 사용하는 함수를 활성화 함수(Activation Function)라고 한다. 
<font color='red'>선형 함수가 아니라 비선형 함수를 사용하는 이유는 딥러닝 모델의 레이어 층을 깊게 가져갈 수 있기 때문이다.</font>  

![image](https://user-images.githubusercontent.com/93988405/230610317-64041e6d-ea3d-418f-abcf-d3e945290a33.png){:class="align-center"}  

![image](https://user-images.githubusercontent.com/93988405/230610372-1dd962b0-49d5-4700-af00-874c2996a40d.png){:class="align-center"}  
![image](https://user-images.githubusercontent.com/93988405/230610382-6b3cdc61-00d5-4b64-b2d1-c19e807bffe6.png){:class="align-center"}  

등의 activation function 들이 있다.  

