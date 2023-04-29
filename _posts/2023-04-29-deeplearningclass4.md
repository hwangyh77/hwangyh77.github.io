---
title:  "Deeplearning class 4"
categories:
  - deeplearning
---

# Generalization  

모델은 Training data를 가지고 모델링을 하는데 모델링의 목적은 training data가 아니라 다른 외부의 data를 모델에 집어넣어도 
training data로 모델을 학습시켜 얻은 accuracy와 비슷한 값을 갖게 하는 것이다.  
학습 데이터와 input data가 달라져도 출력에 대한 성능 차이가 나지 않게 하는 것을 일반화(generalization)라고 한다.  
<font color='red'>따라서 training data로 모델을 학습 시킬 때 정확하게 일반화를 해야 좋은 모델을 얻고, 적용할 수 있다.</font>  

## < Regularization >  

We want to generalize to data it hasn’t seen before.  
<mark style='background-color: #dcffe4'>(Improving generalization = overfitting 방지)</mark>  

이제 Regularization 기법에 어떤것들이 있는지 알아보자!!  

#### - Early stopping  

![image](https://user-images.githubusercontent.com/93988405/235295243-1c5a0ee9-5921-40ed-9542-370ac67c0a65.png){:class="align-center"}  

위의 모델 복잡도 그래프를 보면 특징이 있다.  
그냥 training error는 계속 줄어드는 반면에 validation error를 보면 기울기가 급변하며 하향하다 상향하는 곡선을 그린다.  
따라서 validation error가 감소를 멈추고 증가할 때까지 epoch를 늘려가며 에러가 커지기 시작할 때 멈춘다.  
이를 early stopping이라 부르고 머신러닝에서 neural networks를 학습하는데 널리 쓰이는 알고리즘 중 하나이다.  

하지만 실제는 위의 오른쪽 사진과 같이 변하기 때문에 실제로 사용하기 어렵고 test set을 사용할때는 사용할 수 없다.  
<mark style='background-color: #dcffe4'>(test set을 사용할 때 early stopping을 하는 것 cheating과 같기때문!!)</mark>  

#### - Cross validation  

![image](https://user-images.githubusercontent.com/93988405/235295303-17d18c8b-b109-47c1-8531-e39409c656de.png){:class="align-center"}  

> Validation 은 Model 의 Hyper parameter(사용자가 임의로 지정할 수 있는 Parameter, 예를 들어 Learning rate(학습률) 같은)를 조정하는 Phase이다.  
> Validation data set을 Model에 입력해보며 Hyper parameter를 조정해서 Model의 학습 방향을 정한다고 할 수 있다.  

validation set이 없어도 Training data의 일부를 validation으로 만든다.
Cross validation은 보통 Validation보다 Test의 목적으로 쓰이는 방법론으로, 준비된 Training data가 적을 때 사용한다.  
위의 그림과 같이 validation set을 바꿔가며 실행한다. <font color='red'>이렇게 하면 적은 data set으로 인해 성능 평가의 신뢰도가 떨어지는 문제를 해결할 수 있다.</font>  

#### - L1/L2 Weight decay  

![image](https://user-images.githubusercontent.com/93988405/235295357-e259a4eb-9245-4e27-b74e-6b4097c5ca1d.png){:class="align-center"}  

위와 같은 방식으로 weight에 비례한 값을 만들어 값을 줄일 수 있도록 (완화 시키는)하는 방법이다. 
학습된 모델의 복잡도를 줄이기 위해 학습 중 weight가 너무 큰 값을 갖지 않도록 loss function에 weight가 커질 경우에 대한 패널티 항목(L1/L2)를 넣는다.  

위의 식에서 "람다"는 weight에 얼마나 가중치를 둘지 결정하는 변수이다. 0에 가까울수록 정규화 효과가 없어진다.  

![image](https://user-images.githubusercontent.com/93988405/235295398-c467301e-ecdc-4f7e-b959-f636620e29ec.png){:class="align-center"}  

결국 위의 그래프와 같이 weight를 줄여서 simple 하게 만들어 overfitting을 방지한다.  

#### - Dropout  

![image](https://user-images.githubusercontent.com/93988405/235295417-0d94ed9e-8a33-4e9b-b9ba-b6e36cf27bc5.png){:class="align-center"}  

Dropout은 서로 연결된 연결망(layer)에서 0 ~ 1 사이의 확률로 뉴런을 제거(drop)하는 기법이다.  
이는 가중치가 한쪽으로 몰려서 overfitting 되는 것을 방지하는 앙상블 테크닉의 일종으로 네트워크가 가벼워져서 학습 속도가 향상된다는 장점이 있다.  

위의 사진과 같이 활성화 되는 노드를 랜덤으로 바꿔가며 진행한다.  

#### - Hyperparameter optimization  

> * Parameters in Neural Network  
> 
> Weight & Bias가 있고
> * Hyperparameters in Neural Network  
> 
> learning rate, regularization variable, num of layers, num of hidden units 들이 있다.  

<font color='red'>이러한 값들을 반복해서 경험적으로 찾는데, 이 값들을 잘 찾는 것이 중요하다!!!!</font>  

#### - Data Augmentation  

말 그대로 Training data의 크기를 늘리는 방법이다. 즉 소프트웨어적으로 training data를 늘려 overfitting을 막고 variance를 줄일 수 있는 방법이다.  

![image](https://user-images.githubusercontent.com/93988405/235295488-b5ba0f30-f64d-422a-8211-4aac01e82e01.png){:class="align-center"}  

위와 같이 Rotating the image, flipping, scaling, shifting 등의 기법들을 사용한다.  

#### - Batch Normalization  

![image](https://user-images.githubusercontent.com/93988405/235295514-c6dde2ab-c86d-4474-96c7-d9afd137b637.png){:class="align-center"}  

batch normalization은 학습 과정에서 각 배치 단위별로 데이터가 다양한 분포를 가지더라도 각 배치별로 평균과 분산을 이용해 정규화하는 것을 말한다.  
즉 평균과 분산을 조정하는 과정이 신경망 안에 포함되어 학습할 때 평균과 분산을 조정하는 과정 역시 같이 조절된다.  
위 그림을 보면 batch 단위나 layer에 따라서 입력 값의 분포가 모두 다르지만 정규화를 통하여 분포를 zero mean gaussian 형태로 만들어 준다.  
<font color='red'>Back propagation을 통해 학습한다!!!!</font>
