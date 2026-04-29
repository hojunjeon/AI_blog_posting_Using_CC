---
title: "CNN과 Fine-tuning — 이미지를 '이해'하는 AI 만들기"
tags: [CNN, PyTorch, ResNet, Fine-tuning, 컴퓨터비전, TorchVision]
date: 2026-04-07
---

> **메타 디스크립션**: AI 입문 강의 day5 공부 기록. 컴퓨터 비전의 역사부터 CNN 동작 원리, AlexNet·ResNet 아키텍처, TorchVision 사전학습 모델, Fine-tuning 세 가지 방법까지 — ResNet-18로 분류기를 직접 만든 과정을 정리했습니다.
> **키워드**: CNN 컨볼루션 신경망, ResNet Fine-tuning, TorchVision 사전학습 모델, 컴퓨터비전 입문, Linear Probing
> **예상 읽기 시간**: 10분
> **카테고리**: AI 입문 / 컴퓨터비전 & CNN
> **태그**: CNN, PyTorch, ResNet, Fine-tuning, 컴퓨터비전, TorchVision

---

## 0. 들어가며

AI 강의 day5 내용을 정리한 공부 기록입니다.

day4에서는 PyTorch로 MLP를 구현해 MNIST 손글씨를 분류했습니다. MLP는 이미지를 784개 픽셀의 1차원 벡터로 펼쳐서 입력했는데, 이렇게 하면 픽셀의 위치 정보가 사라집니다. day5에서는 이미지의 2차원 공간 구조를 그대로 활용하는 **CNN(Convolutional Neural Network)** 을 배웠습니다.

커널과 컨볼루션 연산의 원리부터 AlexNet, ResNet 아키텍처, 그리고 사전학습 모델을 재활용하는 **Fine-tuning**까지 한 흐름으로 정리합니다. 실습에서는 ResNet-18 사전학습 모델에 마지막 레이어만 교체해 약 229장의 이미지로 분류기를 만들었습니다.

---

## 1. 컴퓨터 비전이란?

컴퓨터 비전(Computer Vision)은 영상과 이미지를 다루는 전체 분야를 가리킵니다. 이 안에서 접근 방식이 크게 두 가지로 나뉩니다.

| 구분 | 설명 | 특징 |
|------|------|------|
| 전통 영상처리 | 알고리즘·로직으로 이미지를 처리 | 커널 필터, 에지 검출 등 수작업 설계 |
| 딥러닝 기반 | Neural Network으로 이미지를 처리 | 특징 추출 자체를 모델이 학습 |

1970년대에는 연구자가 직접 커널(엣지 검출 필터 등)을 손으로 설계했습니다. "어떤 특징이 중요한가"를 사람이 정해야 했기 때문에, 복잡한 실세계 이미지에서는 한계가 뚜렷했습니다.

전환점은 2012년 **AlexNet**이었습니다. Stanford가 주관하는 ImageNet 대회(1000클래스, 130만 장 이미지)에서 AlexNet이 압도적인 차이로 1위를 차지하면서 딥러닝 기반 컴퓨터 비전의 시대가 열렸습니다. 이후 이미지 인식 분야는 딥러닝이 주도하고 있습니다.

> 💡 **핵심 포인트**: 전통 영상처리는 특징을 사람이 설계하고, 딥러닝은 특징 추출 자체를 모델이 데이터에서 학습합니다.

---

## 2. CNN: 이미지를 '읽는' 방법

### MLP와 CNN의 차이

day4에서 구현한 MLP는 입력 특징을 모두 스칼라 값으로 받았습니다. MNIST에서는 28×28 이미지를 784개 숫자의 1차원 벡터로 펼쳐서 입력했는데, 이 과정에서 **위치 정보가 사라집니다.** "어떤 픽셀이 어떤 픽셀 옆에 있는가"라는 공간 구조를 MLP는 활용하지 못합니다.

CNN의 학습 목표는 다릅니다. **"특징을 잘 뽑아내는 커널(필터)을 구하자"** 가 핵심입니다. 커널을 이미지 위에 슬라이딩하면서 지역적 패턴(경계선, 텍스처, 모양 등)을 추출하기 때문에, 위치 정보를 그대로 반영할 수 있습니다.

### 이미지 텐서 표현

PyTorch에서 이미지는 다음 형태의 텐서로 표현합니다.

```python
# 단일 이미지: (채널, 높이, 너비)
image.shape  # (C, H, W) 예: (3, 224, 224)

# 배치 단위: (배치 크기, 채널, 높이, 너비)
batch.shape  # (N, C, H, W) 예: (32, 3, 224, 224)
```

RGB 이미지라면 채널(C)이 3입니다. 배치로 묶으면 앞에 N이 추가됩니다.

### 커널(Kernel)과 컨볼루션

커널은 작은 크기의 필터(예: 3×3)입니다. 이미지 위에 슬라이딩하면서 겹치는 영역과 내적(dot product)을 계산해 **Feature Map**을 생성합니다.

```python
import torch.nn as nn

# 컨볼루션 레이어: 입력 3채널 → 출력 64채널, 3×3 커널
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
```

- `in_channels`: 입력 이미지 채널 수 (RGB = 3)
- `out_channels`: 출력 Feature Map 채널 수 (커널 개수)
- `padding=1`: 이미지 경계에 0을 추가해 입력 크기 유지

### Conv → ReLU → Pooling 블록

CNN의 기본 빌딩 블록입니다.

```python
# Convolution Block 구조
nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 컨볼루션
    nn.ReLU(),                                    # 비선형 활성화
    nn.MaxPool2d(kernel_size=2, stride=2)         # 공간 크기 절반으로 축소
)
```

- **ReLU**: 커널 적용 후 비선형성 부여
- **MaxPool2d**: Feature Map의 공간 크기를 줄이고(224→112), 가장 두드러진 특징만 남겨 불변성 확보

마지막으로 Feature Map을 1차원으로 펼친(Flatten) 뒤 **Fully Connected Layer(FC)** 로 최종 분류를 수행합니다. CNN에서는 MLP를 "FC Layer"라고 부릅니다.

> 💡 **핵심 포인트**: Conv → ReLU → MaxPool을 여러 번 쌓아 점점 추상적인 특징을 추출하고, 마지막에 Flatten + FC로 분류합니다.

---

## 3. AlexNet에서 ResNet으로

### AlexNet (2012)

AlexNet은 5개의 Convolution Block(Conv + ReLU + MaxPool)과 3개의 FC Layer로 구성된 모델입니다.

```
Conv Block × 5 → Flatten → FC × 3
```

당시 혁신적이었던 이유는 세 가지입니다. GPU를 활용한 대규모 학습, Dropout을 통한 과적합 방지, 그리고 ReLU 활성화 함수 도입입니다. ImageNet 대회에서 2위와 10%p 이상 차이로 1위를 차지하며 CNN 시대를 열었습니다.

다만 한계도 있었습니다. 네트워크를 깊게 쌓을수록 **Vanishing Gradient** 문제가 심해집니다. 역전파 시 gradient가 레이어를 거치면서 점점 작아져 앞쪽 레이어가 제대로 학습되지 않는 현상입니다.

### ResNet (2015, Microsoft)

ResNet은 이 문제를 **잔차 연결(Residual Connection, Skip Connection)** 으로 해결했습니다.

```python
# ResNet의 핵심: 이전 레이어 입력을 출력에 더함
output = F(x) + x   # F(x): Conv Block 출력, x: 입력 그대로
```

이전 레이어의 입력을 다음 레이어 출력에 그대로 더합니다. 덕분에 gradient가 Skip Connection을 통해 앞쪽 레이어까지 직접 흘러갈 수 있습니다. 레이어를 150개 이상 쌓아도 안정적으로 학습이 가능해졌습니다.

| 항목 | AlexNet | ResNet |
|------|---------|--------|
| 출시 | 2012년 | 2015년 |
| 깊이 | 8개 레이어 | 18~152개 레이어 |
| 핵심 아이디어 | GPU 학습, ReLU, Dropout | Residual Connection |
| Vanishing Gradient | 취약 | 해결 |
| 현재 영향력 | CNN 시대 개막 | YOLO, EfficientNet 등에 활용 |

ResNet 이후 대부분의 CNN 기반 모델(YOLO, EfficientNet 등)이 Residual Connection 구조를 채택하고 있습니다.

> 💡 **핵심 포인트**: `output = F(x) + x` — 입력을 출력에 더하는 단순한 아이디어가 Vanishing Gradient 문제를 해결했습니다.

---

## 4. TorchVision으로 사전학습 모델 불러오기

TorchVision은 PyTorch 생태계의 Vision 전용 라이브러리입니다. AlexNet, ResNet, ViT 등 주요 모델의 사전학습 가중치를 바로 불러올 수 있습니다.

```python
import torchvision.models as models
from torchvision import transforms

# ImageNet 사전학습 가중치로 ResNet-18 로드
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
print(model)
# 마지막 FC: Linear(in_features=512, out_features=1000)
```

### ImageNet 정규화

TorchVision 사전학습 모델을 쓸 때는 **ImageNet 통계**로 정규화해야 합니다. 모델이 이 통계로 학습됐기 때문입니다.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet RGB 평균
        std=[0.229, 0.224, 0.225]     # ImageNet RGB 표준편차
    )
])
```

> 💡 **핵심 포인트**: TorchVision 모델은 `weights=` 인자로 사전학습 여부를 지정합니다. `None`이면 랜덤 초기화입니다.

---

## 5. Fine-tuning의 세 가지 방법

Fine-tuning은 **이미 학습된 모델에 새 데이터를 추가해 학습**시키는 기법입니다. 처음부터 학습하는 것보다 훨씬 적은 데이터와 시간으로 좋은 성능을 낼 수 있습니다.

사전학습 모델은 크게 두 부분으로 구분합니다.
- **Backbone**: 특징을 추출하는 Convolution Layer 묶음
- **Head**: 최종 분류를 담당하는 FC Layer

| 방법 | Backbone | Head | 적합한 상황 |
|------|----------|------|------------|
| **Full Fine-tuning** | 전체 파라미터 업데이트 | 업데이트 | 데이터가 충분하고 도메인 차이가 큼 |
| **Partial Fine-tuning** | 일부 레이어만 해제 | 업데이트 | 중간 규모 데이터, 도메인 차이 보정 필요 |
| **Linear Probing** | 완전 동결 (`requires_grad=False`) | Head만 학습 | 데이터 적을 때, 빠른 적응 필요 |

### Linear Probing 코드 예시

```python
# Backbone 동결
for param in model.parameters():
    param.requires_grad = False

# Head(마지막 FC)만 교체 후 학습 허용
model.fc = nn.Linear(512, 2)   # 1000클래스 → 2클래스로 교체
# model.fc는 새로 만들었으므로 requires_grad=True가 기본값
```

`requires_grad=False`로 설정된 파라미터는 `backward()` 호출 시 gradient가 계산되지 않습니다. Backbone의 수백만 개 파라미터를 동결하면 학습 속도가 크게 빨라집니다.

> 💡 **핵심 포인트**: 데이터가 적을 땐 Linear Probing, 데이터가 많을 땐 Full Fine-tuning이 기본 전략입니다.

---

## 6. 실습: ResNet-18 Linear Probing

이번 실습의 목표는 **사전학습된 ResNet-18로 2-클래스 분류기(ronald / others)**를 만드는 것이었습니다. 약 229장의 이미지만으로도 충분한 성능이 나왔습니다.

### 모델 준비

```python
import torchvision.models as models
import torch.nn as nn

# 1. 사전학습 ResNet-18 로드
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 2. Backbone 전체 동결
for param in model.parameters():
    param.requires_grad = False

# 3. 마지막 FC 교체: 1000클래스 → 2클래스
model.fc = nn.Linear(512, 2)

# 학습 가능한 파라미터는 새 FC뿐
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 파라미터 수: {trainable:,}")  # 약 1,026개
```

처음부터 학습할 때와 비교하면 학습 파라미터가 극적으로 줄어듭니다. ResNet-18의 전체 파라미터(약 1100만 개) 중 마지막 FC의 1,026개만 학습합니다.

### 학습 루프

day4에서 익힌 5단계 패턴이 그대로 적용됩니다.

```python
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
```

`optimizer`에 `model.fc.parameters()`만 넘기는 것이 포인트입니다. 동결된 Backbone 파라미터는 optimizer에 포함하지 않아야 합니다.

### 보너스: 데이터 증강 & 학습률 스케줄러

```python
# 데이터 증강: 다양한 변형으로 과적합 방지
train_transform = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 학습률 스케줄러: 일정 epoch마다 lr 감소
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

`RandomHorizontalFlip`은 이미지를 무작위로 좌우 반전합니다. 적은 데이터일수록 증강이 효과적입니다.

---

## 7. Vision Transformer (ViT) 맛보기

강의 말미에 **ViT(Vision Transformer)** 도 잠깐 언급됐습니다.

CNN이 이미지를 작은 커널로 지역적으로 처리하는 반면, ViT는 이미지를 고정 크기의 **패치**로 잘라 Transformer 인코더에 입력합니다. 패치끼리의 관계를 Self-Attention으로 학습하기 때문에, CNN보다 **글로벌한 관계**를 파악하는 데 강점이 있습니다.

TorchVision에서도 `models.vit_b_16()` 형태로 바로 불러올 수 있습니다. Transformer 기반 Vision 모델은 day7 이후 임베딩·Transformer 파트에서 더 자세히 다룰 예정입니다.

---

## 8. 마무리

### 핵심 요약

| 개념 | 설명 |
|------|------|
| 컴퓨터 비전 | 이미지·영상 처리 분야 — 딥러닝 기반이 현재 주류 |
| CNN | 커널로 지역적 특징 추출, 위치 정보 반영 |
| Kernel | 이미지 위를 슬라이딩하며 Feature Map 생성 |
| Padding | 경계에 0 추가, 입력 크기 유지 |
| MaxPooling | Feature Map 공간 축소, 불변성 확보 |
| AlexNet | 2012년 CNN 시대 개막 — GPU, ReLU, Dropout |
| ResNet | Residual Connection으로 Vanishing Gradient 해결 |
| TorchVision | 사전학습 모델·데이터셋·전처리 제공 라이브러리 |
| Fine-tuning | 사전학습 모델을 새 데이터로 추가 학습 |
| Linear Probing | Backbone 동결 + Head만 학습 — 적은 데이터에 효과적 |

### 느낀 점

MLP에서 CNN으로 넘어오면서 "공간 구조를 활용한다"는 말의 의미가 실제로 와닿았습니다. 이미지를 1차원으로 펼치지 않고 2차원 그대로 커널로 훑는다는 게 직관적으로 맞다고 느꼈습니다.

Fine-tuning 실습이 특히 인상적이었습니다. ImageNet 1000클래스를 학습한 모델의 마지막 레이어 하나만 교체했을 뿐인데, 229장의 이미지로도 분류기가 동작하는 걸 보고 "사전학습의 힘"을 체감했습니다. 처음부터 학습하려면 수천 장은 있어야 하는 것을 생각하면 놀라운 효율입니다.

### Day 6 예고

다음은 **파운데이션 모델에서 멀티모달·LLM으로의 진화** 파트입니다. CNN이 이미지 분야를 평정했다면, Transformer와 대규모 사전학습이 언어 분야를 어떻게 바꿨는지, 그리고 이미지와 텍스트를 함께 다루는 멀티모달 모델까지 흐름을 따라갑니다.

이 글이 도움이 됐다면 댓글이나 공감 부탁드립니다. 틀린 내용이 있으면 알려주세요!

---

## 관련 포스트 추천
- [AI 입문] 신경망과 MLP — PyTorch로 XOR부터 MNIST까지 (day4)
- [AI 입문] 파운데이션 모델 → 멀티모달·LLM 진화 (day6)
- ResNet 논문 핵심 정리 — Residual Connection이 왜 효과적인가
- TorchVision Fine-tuning 가이드 — Linear Probing vs Full Fine-tuning

---

#CNN #PyTorch #ResNet #Fine-tuning #컴퓨터비전 #TorchVision #공부기록
