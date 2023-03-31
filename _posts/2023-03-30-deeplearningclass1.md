---
title:  "Deeplearning class 1"
categories:
  - deeplearning
---

# Deeplearning 이란?

딥러닝은 머신러닝의 한 방법으로, 학습 과정 동안 인공 신경망으로서 예시 데이터에서 얻은 일반적인 규칙을 독립적으로 훈련한다. 특히 머신 비전 분야에서 신경망은 일반적으로 데이터와 예제 데이터에 대한 사전 정의된 결과와 같은 지도 학습을 통해 학습된다.  

## 기계학습 V.S 신경망 학습  

#### < 기계학습 >  

학습시킨다 = “적절한 파라미터를 찾는다”  
목표 : Cost function을 최소화 시키기!!!  

#### < 신경망 학습 >  

학습 = 훈련 data로부터 적절한 파라미터(가중치)의 최적값을 자동으로 찾는 것을 의미  
목표 : 마찬가지로 Cost funciton을 최소화 시키기!!!  

같은 데이터 안에서도 데이터를 어떻게 잘라(나눠)서 Train/Test로 나누는지에 따라 모델이 달라진다.  

Train으로 학습을 하고 Validation으로 검증을 하고 Test로 최종성능을 평가한다.  

#### < Train set >  

모델을 학습하기 위한 dataset이다. 한가지 명심해야할 중요한 사실은 
<font color='red'>“모델을 학습하는데에는 오직 유일하게 Train dataset만 이용한다” </font>
보통 Train set을 이용해 각기 다른 모델을 서로 다른 epoch로 학습을 시킨다. 
여기서 다른 모델이란 hidden layer 혹은 hyper parameter에 약간씩 변화를 주는 것을 뜻한다.  


#### < Validation set >  

학습이 이미 완료된 모델을 검증하기 위한 dataset이다. 
<font color='red'>Validation set은 모델을 update, 즉 학습을 시키지는 않지만 학습에 ‘관여’는 한다.</font>  

#### < Test set >  

학습과 검증이 완료된 모델의 성능을 평가하기 위한 dataset이다. 
<font color='red'>Test set은 학습에 전혀 관여하지 않고 오직 ‘최종 성능을 평가’하기 위해 쓰인다.</font>  

보통 Train : Test 데이터를 8 : 2로 나누는데 여기서 Train 데이터 중 일부를 Validation set으로 이용해 결국 Train : Validation : Test를 일반적으로 6 : 2 : 2로 이용한다.  


## Overfitting 과 Underfitting  

- Underfitting : 모델이 너무 간단하기 때문에 학습 오류가 줄어들지 않는 것  

- Overfitting : 학습 오류가 테스트 데이터셋에 대한 오류보다 아주 작은 경우로 모델이 실제 분포보다 학습 샘플들 분포에 더 근접하게 학습되는 현상  

## Bias and Variance  

#### < Bias >  

모델을 통해 얻은 예측값과 실제 정답과의 차이의 평균을 나타낸다. 즉, 예측값이 실제 정답값과 얼마나 떨어져 있는지 나타낸다. -> bias가 높다 : 예측값과 정답 값의 차이가 크다.  

#### < Variance >  

다양한 dataset에 대하여 예측값이 얼마나 변화할 수 있는지에 대한 양(Quantity)의 개념이다. 즉 얼만큼 예측값이 퍼져서 다양하게 출력될 수 있는 정도로 해석할 수 있다.  

![image](https://user-images.githubusercontent.com/93988405/229101930-9e5a856d-6c60-40a2-a629-03e91ac8f92d.png){:class="align-center"}  

> * Bias와 Variance 문제의 정의를 살펴보면 bias 문제는 데이터의 분포에 비하여 모델이 너무 간단한 경우 underfitting이 발생한경우에 발생한다.  
>
> * Variance 문제는 모델의 복잡도가 데이터 분포보다 커서 데이터를 overfitting 시키는 문제를 말한다.


