# Day 7: 토큰화와 임베딩

자연어를 모델 입력으로 변환하는 토큰화(Tokenization)와 임베딩(Embedding)을 학습하고, 시퀀스 모델(RNN/LSTM/Attention/Transformer)의 발전 과정을 이해하는 과정입니다.

---

## 📁 폴더 구조

```
day7_embeddings/
├── 2_1_토큰화_임베딩_(Easy).ipynb   # 메인 강의 노트북
├── 실습/                              # 실습 문제
│   ├── (실습-문제) 2-1_토큰화,임베딩_실습.ipynb
│   └── (실습-힌트) 2-1_토큰화,임베딩_실습.ipynb
└── 과제/                              # 과제
    ├── (과제-문제) 2-1_토큰화,임베딩_심화_과제.ipynb
    └── (과제-힌트) 2-1_토큰화,임베딩_심화_과제.ipynb
```

---

## 📚 주요 학습 내용

### 1. 메인 강의 (`2_1_토큰화_임베딩_(Easy).ipynb`)
- **토크나이저**: 문장을 토큰 단위로 분할하고 수(Index)로 변환하는 도구
- **토큰화 알고리즘**: WordPiece, Byte-Level BPE
- **임베딩**: 토큰 ID를 고정 차원 벡터로 매핑
- **허깅페이스(Hugging Face)**: transformers 라이브러리로 사전 학습 모델 활용

---

## 🧠 토큰화와 임베딩 기본 개념 정리

### 1) 토큰화(Tokenization)
- 텍스트를 모델이 이해할 수 있는 숫자(토큰 ID)로 바꾸는 과정
- 언어별 특성에 따라 토큰화 결과가 달라짐
  - 일반적으로 영어는 1글자당 1byte, 한글은 1글자당 2bytes(UTF-8 기준 3bytes로 처리되는 경우도 있음)
- 모델 성능과 비용에 큰 영향을 주므로, 모델 목적에 맞는 토큰화 방식을 선택하는 것이 중요

#### 대표 토큰화 방식
1. **WordPiece**
   - 단어/서브워드 단위로 토큰화
   - BERT 계열에서 주로 사용
2. **Byte-Level BPE (Byte Pair Encoding)**
   - 바이트 단위로 해석한 뒤, 자주 등장하는 문자열을 병합
   - 다양한 문자 처리에 유연함
3. **SentencePiece**
   - 공백 포함 문자열을 직접 학습해 토큰화(언어 독립적)
   - 문장 단위 처리에 강점이 있어 다국어 모델에서 자주 활용

### 2) 임베딩(Embedding)
- 텍스트(토큰)를 숫자 벡터로 변환하는 과정
- 벡터는 방향과 거리 정보를 가지며, 문장의 의미를 수치 공간으로 표현
- 의미가 비슷한 문장은 벡터 공간에서도 가까워짐
  - 예: `I love AI` 와 `I like artificial intelligence` 는 높은 유사도를 가질 수 있음
- 의미 기반 거리 계산에는 코사인 유사도(cosine similarity) 등을 사용

### 3) 벡터 차원(Dimension)
- 임베딩 벡터 크기 예시: `384`, `768`, `1024`
- 차원이 높을수록 표현력은 좋아질 수 있지만, 연산량/메모리/비용이 함께 증가

### 4) 토큰화 모델 / 임베딩 모델 활용 포인트
- 데이터와 언어 특성에 맞는 토크나이저 선택
- 목표 작업(검색/분류/생성)에 맞는 임베딩 모델 선택
- 정확도와 비용(지연시간, 메모리)을 함께 고려해 운영 기준 수립

### 5) Todo: 임베딩 활용 과제 아이디어
- 유사도 기반 검색(Semantic Search)
- LLM 기반 RAG(Retrieval-Augmented Generation)
- 추천 시스템(벡터 유사도 기반)
- 클러스터링(의미 기반 그룹화)

### 6) 왜 백엔드 서버와 모델 서버를 분리할까?
- 유지보수성과 확장성을 높이기 위해 분리 운영하는 것이 유리
- 모델 교체/배포와 서비스 로직 배포를 독립적으로 진행 가능
- 트래픽 특성(API 요청 vs 추론 요청)에 맞춰 각각 최적화 가능

---

## 🧪 실습 (토큰화·임베딩·시퀀스 모델)

| Step | 주제 | 핵심 내용 |
|------|------|-----------|
| 1 | 토크나이저·임베딩 | NSMC로 WordPiece 학습, 토큰 ID 변환, nn.Embedding 조회 |
| 2 | RNN/LSTM | RNNEncoder·RNNDecoder, Encoder-Decoder 구조, Seq2Seq |
| 3 | Attention | Luong Attention, Score·Context Vector, AttentionDecoder |
| 4 | Huggingface | Helsinki-NLP 번역 모델, Seq2Seq 추론 |
| 5 | 아키텍처별 모델 | BERT(Encoder-only), GPT(Decoder-only) 추론 |

**데이터셋**: NSMC (Naver Sentiment Movie Corpus, 약 20만 리뷰)  
**필요 패키지**: `python>=3.11`, `torch`, `transformers`, `tokenizers`, `sentencepiece`, `sacremoses`, `numpy`, `pandas`

---

## 📝 과제 (토큰화·임베딩 심화)

| Step | 주제 | 핵심 내용 |
|------|------|-----------|
| 1 | 토큰화·임베딩 심화 | 토크나이저 학습, 토큰 ID 시퀀스, nn.Embedding |
| 2 | 시퀀스 모델 | RNN/LSTM Encoder-Decoder, Attention, Seq2Seq |
| 3 | Huggingface 활용 | 사전 학습 모델 추론, BERT·GPT 아키텍처 이해 |

**필요 패키지**: `python>=3.11`, `torch`, `transformers`, `tokenizers`, `numpy`, `pandas`

---

## 🎯 학습 목표

- 토크나이저·토큰화의 개념과 필요성 설명
- WordPiece·BPE 기반 토크나이저 학습 및 사용
- nn.Embedding으로 토큰을 벡터로 변환
- RNN/LSTM의 순차 처리 방식과 Encoder-Decoder 구조 이해
- Attention 메커니즘의 작동 원리와 효과 설명
- BERT(Encoder-only)·GPT(Decoder-only)·Seq2Seq 아키텍처 구분

---

## 📌 참고

- 실습·과제는 TODO 영역을 채우는 형태로 진행
- 강의(토큰화·임베딩) → 실습(Seq2Seq·Attention) → 과제(심화) 순으로 진행 권장
- 허깅페이스 모델 다운로드 시 인터넷 연결 필요
