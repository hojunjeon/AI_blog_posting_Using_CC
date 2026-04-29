---
title: "파운데이션 모델에서 멀티모달까지 — LLM 진화의 흐름 잡기"
tags: [Transformer, LLM, MultiModal, Diffusion, FoundationModel, HuggingFace]
date: 2026-04-07
---

> **메타 디스크립션**: AI 입문 강의 day6 공부 기록. RNN의 한계에서 시작해 Attention, Transformer, Foundation Model, 그리고 텍스트·이미지·음성을 넘나드는 멀티모달 모델까지 — LLM 진화의 전체 흐름을 정리했습니다.
> **키워드**: Transformer Attention 구조, Foundation Model 파운데이션 모델, 멀티모달 LLM, Stable Diffusion 이미지 생성, YOLOS 객체탐지
> **예상 읽기 시간**: 12분
> **카테고리**: AI 입문 / Transformer & LLM
> **태그**: Transformer, LLM, MultiModal, Diffusion, FoundationModel, HuggingFace

---

## 0. 들어가며

AI 강의 day6 내용을 정리한 공부 기록입니다.

day5에서는 CNN이 이미지를 처리하는 원리를 배웠습니다. CNN은 커널로 지역적 패턴을 추출하는 방식이었는데, day6에서는 그보다 훨씬 더 넓은 시야를 가진 **Transformer** 구조와, 그것이 어떻게 오늘날의 GPT·Claude·Gemini 같은 **Foundation Model**로 발전했는지를 배웠습니다.

이번 강의의 흐름은 이렇습니다.

```
RNN (시계열) → Attention 메커니즘 → Transformer
→ Foundation Model (LLM) → 멀티모달 응용 (이미지·음성)
```

한 강의에서 AI의 최근 10년 흐름을 압축해서 보는 느낌이었습니다. 정리합니다.

---

## 1. Transformer 이전: RNN과 그 한계

### 시계열 데이터와 RNN

Transformer가 등장하기 전, 텍스트 같은 **시계열 데이터**는 **RNN(Recurrent Neural Network)** 으로 처리했습니다. RNN은 이전 입력의 Hidden State를 다음 단계로 넘기며 순서 정보를 반영하는 구조입니다.

```
입력: ['안', '녕', '하', '세', '요']
RNN: h₁ → h₂ → h₃ → h₄ → h₅  (hidden state 전달)
```

- **학습** 시에는 `['안', '녕', '하', '세', '요']`를 넣어서, 다음 글자를 예측하도록 학습합니다.
- **추론** 시에는 입력 글자 하나에서 시작해 다음 글자를 하나씩 생성합니다.

### RNN의 세 가지 문제

| 문제 | 설명 |
|------|------|
| **장기 의존성 문제** | 시퀀스가 길어질수록 초반 정보가 Hidden State에서 희석됨 |
| **Vanishing Gradient** | 역전파 시 기울기가 0에 수렴해 앞쪽 레이어 학습 불가 |
| **병렬화 불가** | 앞 토큰 결과가 나와야 다음 토큰 처리 가능 → GPU 활용 비효율 |

LSTM이 장기 의존성·Vanishing Gradient를 어느 정도 완화했지만, 병렬화 문제는 여전히 남았습니다.

---

## 2. Attention과 Transformer

### Attention 메커니즘

Transformer의 핵심 아이디어는 **Attention**입니다. "이 단어는 다른 어떤 단어와 관련이 깊은가?"를 계산해, 각 단어의 표현에 반영하는 방식입니다.

Attention은 세 가지 행렬을 이용합니다.

| 벡터 | 역할 |
|------|------|
| **Query (Q)** | 현재 단어가 다른 단어에 무엇을 물어보는가 |
| **Key (K)** | 각 단어가 어떤 정보를 갖고 있는가 |
| **Value (V)** | 실제로 전달할 정보의 내용 |

Q, K, V 각각은 별도의 Weight 행렬(학습 파라미터)로부터 계산됩니다. Q와 K의 유사도(내적)를 계산해 Attention Score를 구하고, Softmax를 거쳐 V의 가중합을 냅니다.

```python
# Attention 공식 (개념)
scores = Q @ K.T / sqrt(d_k)   # 유사도 계산 후 스케일링
weights = softmax(scores)       # 확률분포로 변환
output = weights @ V            # 가중합으로 최종 표현
```

> 💡 **핵심 포인트**: Attention은 RNN처럼 순서대로 처리하지 않습니다. 모든 토큰을 **동시에** 비교하기 때문에 병렬 처리가 가능하고, 장거리 의존성도 직접 연결됩니다.

### Transformer 구조

Transformer는 **Encoder**와 **Decoder** 두 블록으로 구성됩니다. Encoder와 Decoder 모두 Attention과 MLP(Feed-Forward Network)의 조합입니다.

```
입력 시퀀스 → [Encoder × N] → 문맥 표현
문맥 표현 + 출력 시퀀스 → [Decoder × N] → 다음 토큰 확률
```

- **Encoder**: 입력 전체를 한꺼번에 읽어 의미를 추출합니다.
- **Decoder**: 이미 생성된 토큰들과 Encoder 출력을 받아, 다음 토큰 하나를 예측합니다.

RNN이 해결 못 했던 세 가지 문제가 Transformer에서 해소됩니다.
- Attention으로 **장거리 의존성** 직접 연결
- 병렬 연산으로 **학습 속도** 대폭 향상
- Residual Connection으로 **Vanishing Gradient** 완화

---

## 3. Foundation Model이란?

### 정의

Transformer 시대에 등장한 개념이 **Foundation Model(파운데이션 모델)** 입니다. 세 가지 특징으로 정의할 수 있습니다.

1. **대규모 데이터**로 미리 학습
2. 다양한 태스크에 맞게 **Fine-tuning** 가능
3. **수십억 개** 이상의 파라미터 보유

### 대표 모델

| 구분 | 모델 | 비고 |
|------|------|------|
| 상용 | GPT 시리즈 (OpenAI) | ChatGPT 기반 |
| 상용 | Claude 시리즈 (Anthropic) | |
| 상용 | Gemini (Google) | |
| 오픈소스 | LLaMA (Meta) | 가장 유명한 오픈소스 |
| 오픈소스 | Qwen (Alibaba) | 2025년 4월 Qwen3 출시, 한국어 인식 우수 |
| 오픈소스 | gpt-oss (OpenAI) | 2025년 오픈소스 공개 |

### HuggingFace에서 직접 탐색

실습에서는 [HuggingFace](https://huggingface.co/models)에서 파운데이션 모델을 직접 검색해봤습니다.

- `phi` 검색 → **phi-4 multimodal**: Image + Text 동시 입력 가능한 멀티모달 모델
- `qwen` 검색 → **Qwen-VL**: Vision + Language, 이미지와 텍스트를 함께 입력받는 모델

> 💡 **핵심 포인트**: 모델 이름에 `VL`, `Vision`, `Multimodal` 같은 키워드가 붙으면 이미지도 처리할 수 있는 멀티모달 모델입니다.

---

## 4. 멀티모달(Multimodal)이란?

**모달리티(Modality)** 는 인간이 인식하는 감각 채널을 뜻합니다. Text, Image, Audio, Video, Sensor 데이터 등이 각각 하나의 모달리티입니다. 여러 모달리티를 함께 다루는 모델을 **멀티모달 모델**이라고 합니다.

### Transformer가 멀티모달에 적합한 이유

Transformer는 "시퀀스"만 입력받으면 됩니다.

| 모달리티 | Transformer 입력 변환 방법 |
|----------|--------------------------|
| Text | 단어를 토큰 단위로 분할 |
| Image | 이미지를 고정 크기 **패치(Patch)** 로 잘라 토큰처럼 처리 |
| Audio | 스펙트로그램을 패치로 변환 |

이미지를 패치 단위 토큰으로 바꾸면, 텍스트와 이미지를 같은 공간에서 처리할 수 있습니다. 이를 통해 "어떤 이미지 패치가 어떤 텍스트와 관련이 있는가"를 Attention으로 학습할 수 있습니다.

---

## 5. 실습 1: 객체 탐지 — YOLOS

### 개요

이미지에서 객체의 위치를 박스로 찾아주는 **Object Detection** 태스크를 실습했습니다. CNN 기반의 유명한 모델이 **YOLO(You Only Look Once)** 이고, Transformer 기반으로 대체한 것이 **YOLOS(You Only Look at One Sequence)** 입니다. (이름 센스가...)

HuggingFace의 `hustvl/yolos-small` 모델을 사용했습니다.

```python
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import torch

# 이미지 로드
url = 'https://cdn.pixabay.com/photo/2021/07/09/02/29/family-6398107_1280.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# 모델 로드
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

# 추론
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 결과 시각화 (박스 + 레이블 그리기)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
```

> 💡 CPU에서 약 57초 소요. GPU 환경을 권장합니다.

---

## 6. 실습 2: 이미지 생성 — Stable Diffusion

### Diffusion 모델의 원리

이미지 생성에는 **Diffusion 알고리즘**이 사용됩니다.

- **학습**: 깨끗한 이미지에 노이즈를 점점 추가하면서, "얼마나 노이즈를 추가했는지" 예측하도록 모델을 학습합니다.
- **추론**: 완전한 노이즈(랜덤 픽셀)에서 시작해 노이즈를 조금씩 제거하며 이미지를 생성합니다.

텍스트 조건(프롬프트)을 추가하면 "원하는 내용"을 담은 이미지를 생성할 수 있습니다. 이것이 **Text-to-Image** 모델입니다.

```python
# GPU 필수 (CPU 불가)
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe"
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save("output.png")
```

사용한 모델은 `stabilityai/sd-turbo`로, 단 1~4 Step만으로 이미지를 생성할 수 있는 경량화 버전입니다.

> 💡 **핵심 포인트**: Diffusion 모델은 "노이즈 제거 과정"을 학습하고, 추론 시에는 순수 노이즈에서 역방향으로 이미지를 복원합니다.

---

## 7. 실습 3: 멀티모달 VQA — SmolVLM

### Image-Text to Text

이미지와 텍스트를 함께 입력받아 텍스트로 답하는 **VQA(Visual Question Answering)** 를 실습했습니다. 사용한 모델은 `HuggingFaceTB/SmolVLM-Instruct`입니다.

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

# 두 이미지를 비교 설명 요청
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Can you explain the difference between the two images?"},
        ],
    }
]

inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt").to(DEVICE)
outputs = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(outputs[0]))
```

모델 크기가 약 **4.5GB**로 다운로드 시간이 꽤 걸립니다. 강의에서는 미리 캐시해두고 진행했습니다.

### HuggingFace의 다양한 멀티모달 태스크

HuggingFace Models > Tasks 메뉴에서 멀티모달 모델을 태스크별로 탐색할 수 있습니다.

| 태스크 | 설명 |
|--------|------|
| Text-to-Image | 텍스트 → 이미지 생성 |
| Image-Text-to-Text | 이미지 + 텍스트 → 텍스트 (VQA) |
| Audio-Text-to-Text | 음성 or 텍스트 → 텍스트 (멀티모달) |
| Text-to-Speech | 텍스트 → 음성 합성 |

---

## 8. 마무리

### 핵심 요약

| 개념 | 설명 |
|------|------|
| RNN | 순서대로 처리하는 시계열 모델 — 장기 의존성·병렬화 한계 |
| Attention (Q/K/V) | 토큰 간 관련도를 학습 파라미터로 계산 |
| Transformer | Attention + MLP 조합, 병렬 처리 가능, 멀티모달 확장 용이 |
| Foundation Model | 대규모 사전학습 + Fine-tuning 가능한 대형 모델 |
| 멀티모달 | Text·Image·Audio 등 여러 모달리티를 함께 처리 |
| Diffusion | 노이즈 제거 원리로 이미지 생성 (Stable Diffusion 등) |
| YOLOS | Transformer 기반 Object Detection |
| SmolVLM | Image + Text 동시 입력 → 텍스트 답변 (VQA) |

### 느낀 점

"Transformer가 텍스트만 처리하는 게 아니구나"라는 게 이번 강의의 가장 큰 수확이었습니다. 이미지를 패치로 자르면 그냥 토큰이 되고, Transformer는 그 토큰들을 Attention으로 처리합니다. 텍스트·이미지·음성이 모두 "시퀀스"로 변환되면 동일한 구조로 처리할 수 있다는 발상이 단순하면서도 강력합니다.

Diffusion 모델의 원리도 인상 깊었습니다. "노이즈를 추가하는 것을 역으로 학습한다"는 아이디어가 이렇게 실사 수준의 이미지를 만들어낼 수 있다는 게 놀랍습니다.

### Day 7 예고

다음은 **토큰화 + 임베딩 + Transformer 내부 구조** 파트입니다. day6에서 "Transformer가 텍스트를 어떻게 처리하는가"를 큰 그림으로 봤다면, day7에서는 텍스트가 어떻게 숫자로 변환되고(토큰화), 그 숫자들이 어떤 벡터 공간에서 표현되는지(임베딩)를 더 깊이 파고듭니다.

이 글이 도움이 됐다면 댓글이나 공감 부탁드립니다. 틀린 내용이 있으면 알려주세요!

---

## 관련 포스트 추천
- [AI 입문] CNN과 Fine-tuning — 이미지를 '이해'하는 AI 만들기 (day5)
- [AI 입문] 토큰화·임베딩·Transformer 내부 (day7)
- Attention is All You Need — Transformer 논문 핵심 정리
- HuggingFace 모델 허브 활용 가이드 — 멀티모달 모델 탐색하기

---

#Transformer #LLM #MultiModal #Diffusion #FoundationModel #HuggingFace #공부기록
