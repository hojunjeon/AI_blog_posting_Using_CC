---
title: "토크나이저부터 임베딩까지 — 문장을 AI에게 먹이는 법"
tags: [Tokenizer, Embedding, HuggingFace, BERT, BPE, CosineSimilarity, NLP]
date: 2026-04-07
---

> **메타 디스크립션**: AI 입문 강의 day7 공부 기록. 문장을 숫자로 바꾸는 토크나이저(BERT Word Piece / Byte-Level BPE), 단어에 의미를 담는 임베딩, 코사인 유사도까지 — 텍스트를 AI 모델에 입력하기까지의 전 과정을 정리했습니다.
> **키워드**: 토크나이저, 임베딩 모델, HuggingFace transformers, BERT Word Piece, Byte-Level BPE, 코사인 유사도, Sentence Transformers
> **예상 읽기 시간**: 12분
> **카테고리**: AI 입문 / NLP · 임베딩
> **태그**: Tokenizer, Embedding, HuggingFace, BERT, BPE, CosineSimilarity, NLP

---

## 0. 들어가며

AI 강의 day7 내용을 정리한 공부 기록입니다.

day6에서 Transformer가 어떻게 동작하는지 큰 그림을 배웠다면, day7은 그 모델에 **텍스트를 어떻게 먹이는지**를 다룹니다. 아무리 강력한 모델이 있어도 글자를 그대로 넣을 수는 없습니다. 숫자만 계산할 수 있는 모델을 위해 `"안녕하세요"` 같은 문장을 `[8192, 91352, ...]` 같은 숫자 배열로, 다시 의미가 담긴 벡터로 변환하는 과정이 필요합니다.

이번 강의의 흐름은 이렇습니다.

```
문장 → [토크나이저] → 토큰 ID 리스트
     → [임베딩 모델] → 의미 벡터
     → [Pooling] → 문장 벡터
```

토크나이저와 임베딩. 이 두 가지가 오늘의 핵심입니다.

---

## 1. 토크나이저란?

**토크나이저(Tokenizer)**는 문장을 토큰 단위로 쪼개고, 각 토큰을 숫자(ID)로 바꿔주는 도구입니다.

```
"안녕하세요"
  → 토큰으로 자르기: ["안녕", "하세요"]
  → 숫자(ID)로 변환: [8192, 91352]
```

### 왜 필요할까?

모델의 입력은 항상 숫자입니다 (y = wx + b에서 x처럼). 문장 자체를 통째로 하나의 숫자에 대응시키는 건 불가능합니다. 경우의 수가 무한대이고, 의미가 비슷한 문장도 완전히 다른 숫자가 되어버립니다.

```
"안녕하세요" → 582번
"치킨맛있어" → 582번  ← 이렇게 되면 안됩니다!
```

그래서 문장을 **작은 조각(토큰)**으로 나누고, 각 토큰에 번호를 붙이는 방식을 씁니다. 미리 만들어둔 사전(Vocabulary)을 참고해서 토큰 → ID로 변환하는 거죠.

### 토크나이저 동작 원리

```
입력 문장
    ↓
Vocabulary 참조 (미리 완성된 사전: {단어: Index번호})
    ↓
토큰으로 분해 → ID 리스트로 변환
```

Vocabulary 자체는 거대한 딕셔너리입니다. `klue/bert-base` 기준으로 수만 개의 토큰이 등록되어 있습니다.

---

## 2. 허깅페이스(HuggingFace) 소개

토크나이저를 직접 만들 필요는 없습니다. **허깅페이스(HuggingFace)** 라는 AI 오픈소스 플랫폼에 수십만 개의 모델과 토크나이저가 올라와 있거든요.

- GitHub이 코드 저장소라면, 허깅페이스는 **AI 모델 저장소**입니다.
- `transformers` 라이브러리 한 줄로 원하는 모델을 다운로드하고 바로 쓸 수 있습니다.

```python
from transformers import AutoTokenizer

model_id = "klue/bert-base"  # 한국어에 강한 BERT 기반 모델
tokenizer = AutoTokenizer.from_pretrained(model_id)

korean = "저는 맥도날드에 다녀왔어요."
tokens = tokenizer.tokenize(korean)
ids    = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)  # ['저', '##는', '맥도날드', '##에', '다녀왔어요', '.']
print(ids)     # [2003, 2100, 19865, ...]
```

처음 실행하면 허깅페이스에서 모델 파일을 자동으로 다운로드하고 `~/.cache/huggingface/hub/`에 저장합니다.

---

## 3. 토크나이저 두 종류: Word Piece vs Byte-Level BPE

### BERT 토크나이저 (Word Piece)

BERT 계열 모델이 쓰는 방식입니다. 글자 단위로 쪼갠 뒤, 자주 연속으로 등장하는 글자들을 하나의 토큰으로 묶는 **Word Piece 알고리즘**으로 Vocabulary를 만듭니다.

결과물에서 `##`이 붙은 토큰은 **앞 토큰에 이어지는 접미사**입니다.

```
"저는" → ["저", "##는"]
```

### Byte-Level BPE 토크나이저

GPT2, DeepSeek 등 대규모 LLM에서 주로 쓰이는 방식입니다. 글자 대신 **바이트(Byte) 단위**로 쪼갠 뒤 자주 나오는 바이트 쌍을 합치는 알고리즘으로 Vocabulary를 만듭니다.

```python
model_id = "gpt2"  # Byte-Level BPE 토크나이저
tokenizer = AutoTokenizer.from_pretrained(model_id)

english = "I went to McDonald's."
tokens_eng = tokenizer.tokenize(english)
# ['I', 'Ġwent', 'Ġto', 'ĠMcDonald', "'s", '.']
```

`Ġ`는 **띄어쓰기**를 뜻합니다. 영어는 잘 되지만 한글 지원이 약한 편입니다.

| 구분 | Word Piece (BERT) | Byte-Level BPE (GPT2 등) |
|------|-------------------|--------------------------|
| 단위 | 글자 | 바이트 |
| 접미사 표시 | `##` 접두사 | `Ġ` (띄어쓰기 표시) |
| 대표 모델 | BERT, klue/bert-base | GPT2, DeepSeek |
| 최신 LLM 채택 | 드뭄 | 대부분 |

### 최신 모델 토크나이저 살펴보기

최신 모델인 **DeepSeek**의 토크나이저도 허깅페이스에서 받아볼 수 있습니다. `tokenizer.json`과 `tokenizer_config.json` 두 파일만 있으면 로컬에서 바로 쓸 수 있습니다.

```python
# 로컬 폴더에 tokenizer.json + tokenizer_config.json 저장 후
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_deepseek")
```

---

## 4. 임베딩 모델

### 임베딩이란?

토크나이저가 `"사과"` → `424` 라는 ID를 반환한다면, 임베딩 모델은 그 ID를 **의미 공간의 벡터**로 변환합니다.

```
"사과" → 토큰ID: 424 → 임베딩: [0.21, -0.35, 0.81, ...]
"과일" → 토큰ID: 712 → 임베딩: [0.19, -0.32, 0.79, ...]
"자동차" → 토큰ID: 991 → 임베딩: [-0.53, 0.12, -0.44, ...]
```

**핵심 특징**: 의미가 비슷한 단어는 가까운 벡터값으로 맵핑됩니다. "사과"와 "과일"은 벡터 공간에서 가깝고, "자동차"는 멀리 떨어집니다.

```
토큰 ID만 쓸 때: "사과"=424, "과일"=712 → 완전히 다른 숫자, 의미 연결 없음
임베딩 사용 시: "사과"와 "과일" → 비슷한 방향의 벡터 → 의미 관계 반영
```

### 감정 판별기 예시로 이해하기

임베딩이 왜 필요한지 실감나는 예시입니다.

```
학습 데이터: "나 이 영화 본 것 좋아요" → 긍정
테스트 입력: "저 이 Movie 본 것 좋아요"
```

토큰 ID만 쓴다면 두 문장은 완전히 다른 숫자 배열이라 모델이 헷갈립니다. 하지만 임베딩을 거치면 "나는"과 "저"가 비슷한 벡터로, "Movie"와 "영화"가 비슷한 벡터로 맵핑되어 모델이 제대로 긍정 판별을 할 수 있습니다.

---

## 5. Pooling — 여러 벡터를 하나로

임베딩 모델의 출력은 **토큰 수만큼 벡터**입니다. "사과는 맛나"가 3개 토큰이라면 벡터도 3개.

```
"사과는 맛나"
  → 토크나이저: ["사과", "##는", "맛나"]
  → 임베딩: [[0.2, 0.1, ...], [0.4, -0.3, ...], [0.1, 0.5, ...]]
```

문장 전체를 하나의 벡터로 표현하려면 이 3개를 합쳐야 합니다. 이것을 **Pooling**이라고 합니다.

가장 흔한 방법은 **Mean Pooling** — 모든 토큰 벡터의 평균을 내는 것입니다.

```
문장 벡터 = (토큰1 벡터 + 토큰2 벡터 + 토큰3 벡터) / 3
```

---

## 6. 임베딩 벡터 시각화

`intfloat/e5-small-v2` 모델로 동물·과일·탈것 단어들을 임베딩하고, **PCA로 384차원 → 3차원** 축소 후 3D 산점도로 시각화했습니다.

```python
words = [
    'cat', 'dog', 'tiger', 'lion', 'wolf', 'fox',          # 동물
    'apple', 'banana', 'grape', 'orange', 'peach',          # 과일
    'car', 'bus', 'train', 'bicycle', 'airplane', 'ship'    # 탈것
]

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
model = AutoModel.from_pretrained("intfloat/e5-small-v2")
```

결과를 보면 동물·과일·탈것 그룹끼리 3D 공간에서 확연히 뭉쳐있는 걸 볼 수 있습니다. 임베딩이 실제로 의미를 공간에 담는다는 걸 눈으로 확인하는 순간이었습니다.

---

## 7. 코사인 유사도 (Cosine Similarity)

### 벡터 유사도를 어떻게 측정할까?

임베딩 벡터끼리 얼마나 비슷한지 측정할 때는 **코사인 유사도**를 씁니다. 유클리드 거리(절대 위치)가 아닌, 벡터의 **방향** 차이를 기준으로 하는 방법입니다.

```
코사인 유사도 = (A · B) / (‖A‖ × ‖B‖)
```

- 값 범위: -1 ~ 1 (실제로는 대부분 0.5 ~ 0.9 사이)
- 방향이 같을수록 1에 가까움 → 의미가 유사

### L2 정규화

벡터의 크기를 1로 만드는 **L2 정규화**를 먼저 하면, 코사인 유사도가 단순히 `A · B` (내적) 로 단순해집니다.

```python
# L2 정규화 후 코사인 유사도 계산
import torch.nn.functional as F

embeddings = F.normalize(embeddings, p=2, dim=1)
cosine_sim = embeddings @ embeddings.T
```

| 용도 | L2 정규화 사용 여부 |
|------|-------------------|
| LLM 언어 모델 | 사용 안함 (크기 정보 필요) |
| 이미지/텍스트 검색 | 사용 (유사성 검색, RAG 등) |

---

## 8. Sentence Transformers로 간편하게

임베딩 → Pooling → L2 정규화 → 코사인 유사도 계산을 매번 직접 구현하기 번거롭습니다. **Sentence Transformers** 라이브러리가 이 과정을 자동으로 처리해줍니다.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

sentences = ["나는 맥도날드가 좋다", "I love McDonald's"]
embeddings = model.encode(sentences)
```

Qwen3-Embedding은 Alibaba에서 만든 최신 임베딩 모델로, 한국어도 잘 지원하고 성능 대비 용량이 작아서 인기가 많습니다.

---

## 9. 마무리

### 핵심 요약

| 개념 | 한 줄 설명 |
|------|-----------|
| **토크나이저** | 문장 → 토큰 → ID 변환 도구. Vocabulary 기반 |
| **Word Piece** | BERT 계열. 글자 단위로 쪼개고 접미사에 `##` 표시 |
| **Byte-Level BPE** | GPT2·DeepSeek 계열. 바이트 단위, 최신 LLM 대세 |
| **허깅페이스** | AI 모델 오픈소스 플랫폼. `transformers` 라이브러리로 다운로드 |
| **임베딩 모델** | 토큰 ID → 의미 벡터 변환. 비슷한 의미 = 비슷한 벡터 |
| **Pooling** | 여러 토큰 벡터를 하나의 문장 벡터로 합치기 (주로 Mean Pooling) |
| **코사인 유사도** | 벡터 방향 유사도 측정. 검색·RAG에서 필수 |
| **L2 정규화** | 벡터 크기를 1로 만들어 코사인 유사도 계산 단순화 |
| **Sentence Transformers** | 임베딩 파이프라인(Pooling+정규화+유사도) 자동화 라이브러리 |

### 다음 day 예고

Day 8에서는 **LLM을 이용한 데이터 생성·증강**을 다룹니다. AI로 학습 데이터를 직접 만드는 방법, 즉 Synthetic Data에 대한 이야기입니다.
