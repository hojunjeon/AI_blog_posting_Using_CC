---
title: "모델 양자화 + llama.cpp — CPU에서 LLM 돌리기"
tags: [Quantization, llamacpp, GGUF, Q4KM, TPS, LLMasJudge, FP16, 모델경량화]
date: 2026-04-21
---

> **메타 디스크립션**: AI 입문 강의 day14 공부 기록. 모델 양자화(FP16 vs Q4_K_M)로 메모리 줄이기, GGUF 포맷과 llama.cpp로 CPU 추론하기, TPS 성능 측정, LLM-as-a-Judge로 품질 평가하는 방법을 정리했습니다.
> **키워드**: 양자화, llama.cpp, GGUF, Q4_K_M, TPS, LLM-as-Judge, 모델 경량화
> **예상 읽기 시간**: 11분
> **카테고리**: AI 입문 / 모델 배포
> **태그**: Quantization, llamacpp, GGUF, Q4KM, TPS, LLMasJudge

---

## 0. 들어가며

AI 강의 day14 내용을 정리한 공부 기록입니다.

day12에서 LoRA/QLoRA로 파인튜닝을 했다면, day14는 **만들어진 모델을 어떻게 서비스에 올리느냐**에 관한 이야기입니다. 연구용 추론과 서비스용 추론은 요구사항이 다릅니다.

```
[연구/개발용]  PyTorch + HuggingFace Transformers
  → 유연성 높음, 속도 낮음

[서비스/배포용]  llama.cpp, vLLM
  → 속도 최적화, 메모리 효율 극대화
```

day14의 핵심 흐름:

```
모델 양자화 (FP16 → Q4_K_M)
  → GGUF 포맷으로 변환
  → llama.cpp로 CPU 추론
  → TPS 측정으로 성능 비교
  → LLM-as-Judge로 품질 검증
```

---

## 1. 추론 엔진 — 연구용 vs 서비스용

### 1-1. HuggingFace Transformers (연구용)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

inputs = tokenizer("안녕하세요", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
```

PyTorch 기반이라 디버깅이 쉽고 커스터마이징이 자유롭습니다. 연구 단계에서는 최적 선택이지만, 서비스 환경에서는 속도가 느리고 메모리를 많이 씁니다.

### 1-2. llama.cpp (서비스용)

C++ 기반 추론 엔진입니다. CPU에서도 빠르게 실행되며 메모리 효율이 뛰어납니다.

- CPU만으로 실행 가능 (GPU 없어도 됨)
- Apple Silicon, ARM, x86 모두 지원
- GGUF 포맷 전용

---

## 2. GGUF — llama.cpp 전용 모델 포맷

GGUF(Georgi Gerganov Unified Format)는 llama.cpp 개발자 **Georgi Gerganov**의 이름을 딴 포맷입니다. 양자화된 모델 가중치와 메타데이터를 하나의 파일에 담습니다.

### 2-1. 양자화 레벨

GGUF는 다양한 양자화 수준을 지원합니다.

| 포맷 | 비트수 | 파일 크기 | 품질 |
|------|--------|----------|------|
| FP16 | 16-bit | ~3GB (1.5B) | 최상 |
| Q8_0 | 8-bit  | ~1.5GB | 매우 좋음 |
| Q4_K_M | 4-bit | ~900MB | 좋음 |
| Q2_K | 2-bit  | ~500MB | 저하 |

실습에서는 `Qwen2.5-1.5B` 모델로 FP16과 Q4_K_M을 비교했습니다.

### 2-2. PyTorch 모델을 GGUF로 직접 변환

```bash
# llama.cpp의 변환 스크립트 사용
python llama.cpp/convert_hf_to_gguf.py \
    Qwen/Qwen2.5-1.5B \
    --outfile qwen2.5-1.5b-fp16.gguf \
    --outtype f16
```

파인튜닝한 로컬 모델도 같은 방식으로 GGUF로 변환해서 배포할 수 있습니다.

---

## 3. TPS — 추론 속도 측정

TPS(Tokens Per Second)는 LLM 추론 성능의 핵심 지표입니다. 1초에 몇 개의 토큰을 생성하는지를 나타냅니다.

### 3-1. llama.cpp로 TPS 측정

```python
from llama_cpp import Llama

# FP16 모델 로드
llm_fp16 = Llama(model_path="./qwen2.5-1.5b-fp16.gguf", n_ctx=512)

import time

start = time.time()
response = llm_fp16("안녕하세요, 자기소개를 해주세요.", max_tokens=200)
elapsed = time.time() - start

tokens_generated = response["usage"]["completion_tokens"]
tps = tokens_generated / elapsed
print(f"FP16 TPS: {tps:.1f} tokens/sec")
```

### 3-2. FP16 vs Q4_K_M 비교 결과

실습에서 측정한 결과입니다.

| 모델 | 파일 크기 | TPS (CPU) | 메모리 |
|------|----------|-----------|--------|
| FP16 | ~3.0GB | ~8 tok/s | ~4GB |
| Q4_K_M | ~0.9GB | ~25 tok/s | ~1.5GB |

Q4_K_M은 FP16 대비 파일 크기가 **70% 감소**, 속도는 **3배 향상**됩니다. CPU만 있는 환경에서 실시간 응답을 원한다면 Q4_K_M이 실용적인 선택입니다.

---

## 4. LLM-as-a-Judge — 양자화 품질 자동 평가

속도는 측정했는데, 양자화로 줄어든 **답변 품질**은 어떻게 평가할까요? 사람이 일일이 읽고 평가하는 건 비효율적입니다. 여기서 다시 **LLM-as-a-Judge** 방식을 활용합니다.

```python
def judge_quality(original_answer: str, quantized_answer: str, question: str) -> dict:
    judge_prompt = f"""
당신은 두 AI 답변의 품질을 공정하게 평가하는 심사자입니다.

질문: {question}

답변 A (고품질 기준):
{original_answer}

답변 B (평가 대상):
{quantized_answer}

다음 기준으로 답변 B를 1~5점으로 평가해주세요:
1. 정확성: 답변 A와 같은 정보를 담고 있는가
2. 자연스러움: 문장이 자연스럽고 읽기 좋은가
3. 완전성: 중요한 내용이 빠지지 않았는가

각 항목 점수와 이유를 JSON 형식으로 답해주세요.
    """
    
    response = judge_model.invoke(judge_prompt)
    return parse_scores(response.content)
```

FP16 모델의 답변을 **정답(Gold Standard)**으로, Q4_K_M 모델의 답변을 평가 대상으로 삼습니다. LLM-as-Judge로 측정한 결과, Q4_K_M은 FP16 대비 품질이 약 0.3~0.5점 낮았습니다 — 일반 사용 목적에서는 체감하기 어려운 수준입니다.

---

## 5. 양자화 선택 기준

어떤 양자화를 써야 할까요?

| 상황 | 추천 포맷 |
|------|----------|
| 연구/실험 | FP16 (품질 최우선) |
| 일반 서비스 | Q4_K_M (속도/품질 균형) |
| 엣지 디바이스 (라즈베리파이 등) | Q2_K (메모리 극소화) |
| GPU 있음 + 빠른 속도 | Q8_0 |

실무에서는 Q4_K_M이 가장 흔히 쓰이는 선택입니다. 품질 저하가 적고, 크기와 속도 개선이 크기 때문입니다.

---

## 6. 마무리

### 핵심 개념 요약

| 개념 | 설명 | 포인트 |
|------|------|--------|
| **추론 엔진** | PyTorch vs llama.cpp | 연구용 vs 서비스용 |
| **GGUF** | llama.cpp 전용 포맷 | 양자화 가중치 + 메타데이터 |
| **Q4_K_M** | 4-bit 양자화 | 속도 3배↑, 크기 70%↓ |
| **TPS** | 초당 생성 토큰 수 | 추론 성능 핵심 지표 |
| **LLM-as-Judge** | LLM으로 품질 자동 평가 | FP16 기준으로 비교 |
| **GGUF 변환** | `convert_hf_to_gguf.py` | 로컬 모델도 변환 가능 |

day14의 핵심은 **모델을 만드는 것에서 배포하는 것**으로의 관점 이동입니다. 아무리 좋은 모델도 실제 서비스에서 빠르고 안정적으로 동작하지 않으면 의미가 없습니다. 양자화와 llama.cpp는 제한된 환경에서 LLM을 실용적으로 쓰기 위한 핵심 도구입니다.

---

**강의 마무리**

day1부터 day14까지, Python 기초부터 시작해서 LLM 배포까지 왔습니다. 이번 시리즈에서 다룬 흐름을 돌아보면:

```
Python/NumPy → 선형회귀 → EDA → 신경망/MLP → CNN
→ LLM 이해 → 임베딩/Transformer → 데이터 생성
→ LangChain/RAG → AI Agent → PEFT/LoRA → 양자화 배포
```

입문 강의이지만, 실제로 AI 서비스를 만드는 데 필요한 전 과정을 체험할 수 있었습니다.

---

## 관련 포스트 추천
- Day 12 - PEFT · LoRA · QLoRA: 효율적인 파인튜닝
- Day 8 - LLM으로 학습 데이터 만들기
