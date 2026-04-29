---
title: [AI 입문] 신경망(Neural Network)과 MLP — PyTorch로 XOR부터 MNIST까지
tags: PyTorch, MLP, 신경망, Autograd, MNIST, 공부기록
date: 2026-04-07
---

> **메타 디스크립션**: AI 입문 강의 day4 공부 기록. 선형의 한계(XOR 문제)에서 출발해 PyTorch로 Tensor, Autograd, nn.Linear, Sequential, MLP를 직접 구현하는 과정을 정리했습니다.
> **키워드**: PyTorch 입문, MLP 다층퍼셉트론, 신경망 구현, Autograd 자동미분, MNIST 분류
> **예상 읽기 시간**: 9분
> **카테고리**: AI 입문 / 신경망 & PyTorch
> **태그**: PyTorch, MLP, 신경망, Autograd, MNIST, 공부기록

---

## 0. 들어가며

AI 강의 day4 내용을 정리한 공부 기록입니다.

day2에서 NumPy로 경사하강법을 직접 짰고, day3에서 Scikit-learn 파이프라인을 익혔습니다. day4는 본격적으로 **딥러닝**으로 넘어가는 파트입니다. 왜 신경망이 필요한지, PyTorch가 무엇인지부터 시작해서 MNIST 손글씨 분류까지 한 번에 흐름을 잡았습니다.

처음 PyTorch를 접하면 `requires_grad`, `backward()`, `zero_grad()` 같은 개념이 낯설 수 있는데, 학습 루프 순서를 패턴으로 외워두면 훨씬 편합니다. 이 글도 코드 중심으로 정리해 둡니다.

---

## 1. 선형의 한계 — 왜 신경망이 필요한가

### y = ax + b의 한계

day2에서 구현한 선형 분류기는 결국 `y = ax + b` 형태의 직선 하나로 데이터를 나눕니다. 직선으로 나눌 수 있는 문제(선형 분리 가능)라면 완벽하지만, 그렇지 않은 경우엔 아무리 학습해도 한계가 있습니다.

### XOR 문제

대표적인 예가 **XOR 문제**입니다.

| 입력 A | 입력 B | XOR 출력 |
|--------|--------|----------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR은 직선 하나로 0과 1을 분리할 수 없습니다. 1970년대 AI 연구자들이 이 문제를 해결하지 못해 연구비가 끊기고 AI 겨울이 찾아왔습니다.

해결책은 **함수를 여러 개 합성**하는 것입니다. 1986년 제프리 힌튼 교수가 역전파(Backpropagation)와 다층 신경망을 결합하면서 XOR 문제를 풀었고, 딥러닝의 기초가 마련됩니다.

> 💡 **핵심 포인트**: 선형 레이어를 아무리 쌓아도 결국 선형입니다. 비선형 표현을 위해 **활성화 함수**가 필요합니다.

---

## 2. PyTorch 소개

PyTorch는 Meta(구 Facebook)에서 만든 딥러닝 전용 오픈소스 라이브러리입니다.

| 항목 | PyTorch | TensorFlow |
|------|---------|------------|
| 문법 스타일 | 파이썬 친화적, 직관적 | 상대적으로 복잡 |
| 주요 사용층 | AI 연구자 | 프로덕션 |
| 기본 데이터 구조 | Tensor | Tensor |

강의에서 강조한 건 PyTorch가 "더 파이써닉하다"는 점입니다. NumPy 배열을 다루던 감각으로 자연스럽게 넘어올 수 있습니다.

---

## 3. Tensor — PyTorch의 기본 단위

PyTorch에서는 NumPy `ndarray` 대신 **Tensor**를 씁니다. 구조는 거의 동일하지만 GPU 연산을 지원한다는 차이가 있습니다.

```python
import torch

# 1차원 벡터
a = torch.tensor([1, 2, 3])

# 2차원 행렬
b = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(a.shape)  # torch.Size([3])
print(b.shape)  # torch.Size([2, 3])
```

- **1차원**: 벡터
- **2차원**: 행렬
- **N차원**: 텐서

NumPy에서 쓰던 슬라이싱, 연산자 오버로딩, shape 확인 등 대부분의 문법이 그대로 동작합니다.

---

## 4. Autograd — 자동 미분

딥러닝 학습의 핵심은 **역전파(Backpropagation)** 입니다. 미분을 직접 계산하는 게 아니라, PyTorch가 자동으로 gradient를 계산해 줍니다.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1

y.backward()          # 역전파 실행
print(x.grad)         # dy/dx = 2*2 + 3 = 7.0
```

- `requires_grad=True`: 이 Tensor의 미분을 추적하겠다는 선언
- `y.backward()`: 역전파 실행 — gradient 계산
- `x.grad`: 계산된 gradient 값 확인

### 학습 시 vs. 검증/테스트 시

```python
# 학습 시 (gradient 추적 ON)
output = model(input)

# 검증 / 테스트 시 (gradient 추적 OFF — 메모리 절약)
with torch.no_grad():
    output = model(input)
```

`torch.no_grad()`는 gradient 계산을 끄는 컨텍스트 매니저입니다. 검증·테스트 단계에서는 역전파가 필요 없으므로 반드시 꺼서 메모리를 아껴야 합니다.

> 💡 **핵심 포인트**: Autograd는 학습 때는 ON, 검증/테스트 때는 `torch.no_grad()`로 OFF합니다.

---

## 5. nn.Linear — 선형 레이어

`y = ax + b`를 PyTorch로 표현하면 `nn.Linear`입니다. 수식에서 `a`는 Weight, `b`는 Bias에 해당합니다.

```python
import torch.nn as nn

# 입력 1개 → 출력 1개인 선형 레이어
linear = nn.Linear(in_features=1, out_features=1)

# tips 데이터로 선형회귀: total_bill → tip
# 입력 특성 1개, 출력 1개
model = nn.Linear(1, 1)
```

PyTorch의 선형 레이어는 Weight와 Bias를 자동으로 초기화하고, 학습 중에 `optimizer.step()`으로 업데이트합니다. day2에서 NumPy로 직접 관리하던 파라미터를 이제 PyTorch가 자동으로 처리해 줍니다.

---

## 6. Sequential — 레이어 쌓기

수학에서 함수 합성 `f(g(x))`처럼, 여러 레이어를 순서대로 연결하는 구조가 `nn.Sequential`입니다.

```python
# 로지스틱 회귀: y = sigmoid(ax + b)
logistic_model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)
```

Sequential의 진짜 힘은 여기에 **활성화 함수를 끼워 넣을 수 있다**는 것입니다.

### 왜 활성화 함수가 필요한가

선형 레이어만 쌓으면 어떻게 될까요?

```
y = W2 * (W1 * x + b1) + b2
  = (W2*W1)*x + (W2*b1 + b2)
  = W'*x + b'     ← 결국 선형
```

레이어를 100개 쌓아도 결국 하나의 선형 함수입니다. 비선형 표현을 위해 레이어 사이에 **ReLU** 같은 활성화 함수를 끼워 넣어야 합니다.

### ReLU

```python
# ReLU: 음수는 0, 양수는 그대로
relu = nn.ReLU()
print(relu(torch.tensor([-2.0, 0.0, 3.0])))
# tensor([0., 0., 3.])
```

ReLU는 단순하지만 강력합니다. 2010년대 딥러닝 르네상스의 주역 중 하나입니다. 이전에 쓰이던 sigmoid/tanh와 달리 gradient vanishing 문제가 적어서 깊은 네트워크에서 안정적으로 작동합니다.

> 💡 **핵심 포인트**: Linear → ReLU → Linear → ReLU → ... 이 패턴이 MLP의 기본 구조입니다.

---

## 7. MLP (Multi-Layer Perceptron) — 퍼셉트론의 역사

### 퍼셉트론의 역사

| 연도 | 사건 |
|------|------|
| 1958년 | 퍼셉트론 발명 — 단일 선형함수 + 활성화 |
| 1969년 | XOR 문제로 한계 드러남 → AI 겨울 |
| 1986년 | 역전파 + 다층 신경망으로 XOR 해결 (제프리 힌튼) |
| 2010년 | ReLU + GPU → 딥러닝 르네상스 |

### 표현학습 (Representation Learning)

신경망 이전에는 연구자가 직접 "어떤 특징을 뽑을지" 설계해야 했습니다. 표현학습은 특징 추출 자체를 신경망이 스스로 배우는 패러다임입니다.

- **CNN**: 이미지를 그대로 입력 → 신경망이 경계선, 텍스처, 모양 등 특징을 자동 추출
- **RNN**: 시퀀스를 그대로 입력 → 시간적 패턴을 자동 추출

---

## 8. MNIST 손글씨 분류 — 3-Layer MLP

### 구조 설계

MNIST는 0~9 손글씨 숫자 이미지(28×28 픽셀) 데이터셋입니다.

```
입력: 28×28 = 784개 픽셀
출력: 0~9 중 하나 = 10개 클래스

784 → 256 → 128 → 10
```

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

마지막 레이어에는 활성화 함수를 붙이지 않습니다. `CrossEntropyLoss`가 내부적으로 Softmax를 포함하기 때문입니다.

### 손실 함수 — CrossEntropyLoss

분류 문제에는 **Cross Entropy Loss**를 씁니다.

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
```

회귀 문제의 MSE와 달리, 분류에서는 모델이 정답 클래스에 얼마나 높은 확률을 부여했는지를 기준으로 손실을 계산합니다.

이 구조로 학습하면 **90% 이상의 정확도**를 달성할 수 있습니다.

---

## 9. DataLoader — 배치 학습

실제 학습에서는 전체 데이터를 한 번에 넣지 않습니다. **배치(batch)** 단위로 나눠서 넣는 것이 일반적입니다.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True    # 매 epoch마다 순서 랜덤화
)
```

- `batch_size`: 한 번에 모델에 입력하는 데이터 수
- `shuffle=True`: 매 epoch마다 데이터 순서를 섞음 — 모델이 순서를 외우는 것 방지

배치 단위 학습은 메모리 효율도 좋고, gradient 업데이트가 더 자주 일어나기 때문에 수렴 속도도 빠릅니다.

---

## 10. 학습 루프 — 5단계 패턴

PyTorch의 학습 루프는 매번 같은 순서를 따릅니다. 이 순서를 패턴으로 외워두면 어떤 모델을 짜도 그대로 적용할 수 있습니다.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for images, labels in train_loader:
        images = images.view(-1, 784)           # (batch, 28, 28) → (batch, 784)

        optimizer.zero_grad()                   # 1. gradient 초기화
        output = model(images)                  # 2. 순전파
        loss = criterion(output, labels)        # 3. 손실 계산
        loss.backward()                         # 4. 역전파
        optimizer.step()                        # 5. 파라미터 업데이트
```

### 5단계 학습 루프 요약

| 순서 | 코드 | 역할 |
|------|------|------|
| 1 | `optimizer.zero_grad()` | 이전 gradient 초기화 (누적 방지) |
| 2 | `output = model(input)` | 순전파 — 예측값 계산 |
| 3 | `loss = criterion(output, target)` | 손실 계산 |
| 4 | `loss.backward()` | 역전파 — gradient 계산 |
| 5 | `optimizer.step()` | gradient로 파라미터 업데이트 |

`optimizer.zero_grad()`를 맨 앞에 해야 하는 이유가 중요합니다. PyTorch는 `backward()`를 호출할 때마다 gradient를 **누적**하기 때문에, 초기화하지 않으면 이전 배치의 gradient가 남아 있어 잘못된 방향으로 학습됩니다.

> 💡 **핵심 포인트**: `zero_grad() → 순전파 → 손실 → backward → step` 이 순서는 PyTorch 학습의 기본 패턴입니다.

---

## 11. 마무리

### 핵심 요약

| 개념 | 설명 |
|------|------|
| XOR 문제 | 선형 분류기의 한계 — 신경망 필요성 |
| Tensor | PyTorch의 기본 자료구조, GPU 지원 |
| Autograd | `requires_grad=True` + `backward()` → 자동 미분 |
| nn.Linear | `y = Wx + b` 선형 레이어 |
| nn.Sequential | 레이어 순서대로 연결 |
| ReLU | 비선형 활성화 함수 — 음수 0, 양수 그대로 |
| MLP | 선형 + ReLU를 여러 겹 쌓은 다층 신경망 |
| DataLoader | 배치 단위로 데이터 공급, shuffle 지원 |
| 학습 루프 | zero_grad → 순전파 → 손실 → backward → step |

### 느낀 점

day2에서 NumPy로 경사하강법을 손으로 짰을 때 "이걸 실제로 매번 직접 구현하나?" 싶었는데, PyTorch를 쓰니 역전파와 gradient 계산을 Autograd가 다 해줘서 로직 자체에 집중할 수 있었습니다. 학습 루프의 5단계 순서는 처음에는 "왜 이 순서여야 하는가"가 잘 와닿지 않았는데, `zero_grad()`를 빠트렸을 때 gradient가 누적되는 현상을 보고 나서 이해됐습니다.

XOR 문제와 AI 겨울의 역사를 짚고 가는 것도 좋았습니다. 단순히 코드만 배우는 게 아니라 "왜 이게 필요했는가"의 맥락을 알면 개념이 훨씬 자연스럽게 연결됩니다.

### day5 예고

다음은 **CNN(Convolutional Neural Network)과 Fine-tuning** 파트입니다. MLP는 이미지를 784개 픽셀의 1차원 벡터로 펼쳐서 입력했는데, CNN은 이미지의 2차원 공간 구조를 그대로 활용합니다. 컨볼루션 연산이 왜 이미지에 강한지, Fine-tuning으로 사전학습된 모델을 어떻게 활용하는지를 다룹니다.

이 글이 도움이 됐다면 댓글이나 공감 부탁드립니다. 틀린 내용이 있으면 알려주세요!

---

## 관련 포스트 추천
- [AI 입문] EDA + Scikit-learn — 데이터 탐색부터 ML 파이프라인까지 (day3)
- [AI 입문] CNN과 Fine-tuning — 이미지 인식의 핵심 (day5)
- PyTorch 기초 — Tensor와 Autograd 완전 정복
- MLP로 MNIST 분류하기 — 3-Layer 신경망 구현

---

#PyTorch #MLP #신경망 #Autograd #MNIST #공부기록
