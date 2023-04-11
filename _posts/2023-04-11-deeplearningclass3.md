---
title:  "Deeplearning class 3"
categories:
  - deeplearning
---

# Learning Neural Nets  

이번에는 backpropagation과 gradient descent에 대해 공부해 보았다.  

## < Backpropagation >  

Backpropagation은 Artificial Neural Network를 학습시키기 위한 알고리즘 중 하나이다. 
역전파라고 해석되는데, 내가 뽑고자 하는 target 값과 실제 모델이 계산한 output이 얼마나 차이가 나는지 구한 후 
그 오차값을 다시 뒤로 전파해가면서 각 노드가 가지고 있는 변수들을 갱신하는 알고리즘이다.  

Backpropagation의 미분값들을 gradient descent에 사용한다. 
먼저 Gradient Descent에 대해 알아보자!!  

## < Gradient Descent >  

![image](https://user-images.githubusercontent.com/93988405/231022026-01509ca8-af83-4309-b576-59249d2e2ba7.png){:class="align-center"}  

미분을 통해 특정 지점에서의 순간 변화율이 0인 지점을 찾는다면 그 부분이 모델에서 성능이 가장 좋은 지점이다.
따라서 기울기가 가파르지 않은 방향으로 계속 나아가며 순간 변화율이 0인 지점을 찾아가는 과정을 “경사 하강법”이라 한다.  

그저 미분계수가 0인 지점을 찾는게 아닌 gradient descent를 사용해 함수의 최소값을 찾는 이유는 
우리가 실제 분석에서 보는 함수들은 닫힌 형태가 아니거나 함수의 형태가 복잡하여 미분계수와 그 근을 계산하기 어렵고, 
gradient descent가 컴퓨터로 쉽게 구현할 수 있기 때문이다. 
또한 데이터양이 매우 큰 경우 gradient descent와 같은 iterative한 방법을 통해 해를 구하면 계산량 측면에서 더 효율적이라 한다.  

![image](https://user-images.githubusercontent.com/93988405/231022140-fcb05c3d-e5e2-4478-a00c-94c7d62481af.png){:class="align-center"}  

![image](https://user-images.githubusercontent.com/93988405/231022181-1015c98a-e8f5-4884-96bb-25eb83373f3f.png){:class="align-center"}  

learning rate가 커질수록 w의 변화가 빨라진다.  

## < Backpropagation >  

다시 backpropagation을 보면, Backpropagation은 Chain rule을 사용한다.

> chain rule  
> ![image](https://user-images.githubusercontent.com/93988405/231022290-f74a6fba-6a3d-499c-8786-6de93b1a9c94.png)  
> 두 개의 함수 f와 g가 있고 이것들로 이뤄진 복합함수에 대해 편미분을 구하면 위의 공식과 같이 chain rule이 성립한다.  


![image](https://user-images.githubusercontent.com/93988405/231022345-28b0ffbc-0934-41ef-b826-bf15ec492b52.png){:class="align-center"}  

<mark style='background-color: #dcffe4'>계산하나 끝난 것을 여러 군데에서 사용한다 -> chain rule = “효율적 계산”</mark>  

> <순방향 전파(Forward Pass)>  
> ![image](https://user-images.githubusercontent.com/93988405/231022477-ce2d8760-096a-43b0-882f-d802d1ca5e55.png){:class="align-center"}  
> 
> <역전파 Backpropagation>  
>
>![image](https://user-images.githubusercontent.com/93988405/231022527-056e7ebb-4fc6-463d-af41-787a7568b340.png){:class="align-center"}  
>![image](https://user-images.githubusercontent.com/93988405/231022555-10c05221-e73a-4ea9-abc0-103632a4931a.png){:class="align-center"}  


## < Vanishing gradient problem >  

신경망의 모형은 보이는 층(Visible layer)과 숨겨진 층(Hidden layer)로 구성되어 있다. 
보이는 층은 입력층(Input layer)과 출력층(Output layer)으로 구성되어 있고 그 안에서 어떤 계산이 이루어지는지 볼 수 없기때문에 숨겨진 층, 또는 은닉층이라고 부른다.  
은닉층이 많은 다층 퍼셉트론에서, 은닉층을 많이 거칠수록 전달되는 오차가 크게 줄어들어 학습이 되지 않는 현상이 발생하는데, 이를 기울기 소멸 문제라고 한다. 
즉 <font color='red'>중간에 기울기가 0이 되면 뒤에가 다 0이 되어버리는 문제!!</font>  

이를 해결하기 위한 방법으로 사라져가는 성질을 갖지 않는 비선형 함수를 활성화 함수(예 : ReLu함수)로 선택하면 해결할 수 있다.  
> < ReLu >  
> ![image](https://user-images.githubusercontent.com/93988405/231022797-e31d4ea6-dd9c-4f2a-b6bf-d09d8e51f65e.png)
