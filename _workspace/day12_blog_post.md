---
title: "LoRA · QLoRA · Unsloth — GPU 없이도 LLM 파인튜닝 하는 법"
tags: [PEFT, LoRA, QLoRA, Unsloth, FineTuning, LLM, Gemma, Qwen, ChatTemplate]
date: 2026-04-16
---

> **메타 디스크립션**: AI 입문 강의 day12 공부 기록. LLM 파인튜닝의 종류(Full/Partial/PEFT), LoRA의 수학적 원리(저랭크 행렬 분해), QLoRA와 4-bit 양자화, Unsloth로 2배 빠른 학습까지 정리했습니다.
> **키워드**: PEFT, LoRA, QLoRA, Unsloth, 파인튜닝, LLM, 저랭크 행렬
> **예상 읽기 시간**: 13분
> **카테고리**: AI 입문 / 모델 학습
> **태그**: PEFT, LoRA, QLoRA, Unsloth, FineTuning, LLM

---

## 0. 들어가며

AI 강의 day12 내용을 정리한 공부 기록입니다.

LLM을 그냥 쓰는 것과, 내 목적에 맞게 **파인튜닝해서 쓰는 것**은 차원이 다릅니다. 문제는 LLM을 파인튜닝하려면 어마어마한 GPU 메모리가 필요하다는 점입니다. Llama 3 같은 모델을 Full Fine-tuning하면 수십 GB의 VRAM이 필요해서 일반 개발자가 접근하기 어렵습니다.

day12에서는 이 문제를 해결하는 기법들을 다룹니다.

```
Full Fine-tuning (너무 비쌈)
  → Partial Fine-tuning (그래도 비쌈)
  → PEFT / LoRA (훨씬 효율적!)
  → QLoRA (4-bit 양자화 + LoRA = 극한 절약)
  → Unsloth (2배 빠르게, 70% 메모리 절약)
```

---

## 1. 파인튜닝의 종류

### 1-1. Full Fine-tuning

모델의 모든 파라미터를 업데이트합니다. 성능은 가장 좋지만 비용도 가장 큽니다.

- 7B 모델: ~28GB VRAM 필요 (A100 1장도 빠듯)
- 비용: GPU 비용 폭탄 각오

### 1-2. Partial Fine-tuning

특정 레이어만 업데이트합니다. 마지막 레이어나 특정 블록만 학습시키는 방식입니다. Full보다 효율적이지만 여전히 메모리를 많이 씁니다.

### 1-3. PEFT (Parameter-Efficient Fine-Tuning)

기존 모델 파라미터는 **완전히 고정**하고, 추가로 삽입한 **소수의 파라미터만** 학습합니다. 성능은 Full Fine-tuning에 근접하면서 비용은 수십 배 저렴합니다. LoRA가 대표적인 PEFT 기법입니다.

---

## 2. LoRA — 저랭크 행렬로 파라미터 줄이기

LoRA(Low-Rank Adaptation)의 핵심 아이디어는 이렇습니다.

### 2-1. 수학적 원리

Transformer의 가중치 행렬 W를 직접 수정하는 대신, **변화량 ΔW를 두 개의 작은 행렬 A × B로 근사**합니다.

```
W_new = W + ΔW
ΔW ≈ A × B

여기서:
  W: N × N 행렬 (원본, 학습 안 함)
  A: N × r 행렬 (학습 대상)
  B: r × N 행렬 (학습 대상)
  r: rank (보통 8~64, 매우 작은 값)
```

예를 들어 N=1000, r=8이라면:

| 방법 | 파라미터 수 |
|------|------------|
| W 직접 학습 | 1,000 × 1,000 = **100만** |
| LoRA (A + B) | 1,000 × 8 + 8 × 1,000 = **1만 6천** |

같은 모델에서 학습 파라미터가 **약 62배** 줄어듭니다. 이게 LoRA의 마법입니다.

### 2-2. LoRA 적용

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,              # rank (작을수록 파라미터 ↓, 표현력 ↓)
    lora_alpha=16,    # 학습률 스케일링 계수
    target_modules=["q_proj", "v_proj"],  # 적용할 레이어
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 예: trainable params: 2,097,152 || all params: 3,728,893,952 || trainable%: 0.056%
```

전체 파라미터의 0.05% 정도만 학습합니다. 나머지는 고정이라 메모리 사용량이 크게 줄어듭니다.

---

## 3. QLoRA — 4-bit 양자화 + LoRA

QLoRA는 LoRA에서 한 발 더 나아갑니다. 모델 가중치를 **4-bit로 양자화**해서 메모리를 극단적으로 줄인 다음, 그 위에 LoRA를 적용합니다.

```
FP32 (32-bit) 모델: 28GB
FP16 (16-bit) 모델: 14GB
Q4  (4-bit)  모델: ~3.5GB  ← QLoRA 기준점
```

4-bit 양자화 단계에서 모델을 불러올 때 설정합니다.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # NormalFloat4
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

16GB VRAM GPU에서도 70B 급 모델을 QLoRA로 파인튜닝할 수 있습니다.

---

## 4. Unsloth — 2배 빠르게, 70% 메모리 절약

Unsloth는 LoRA/QLoRA 파인튜닝을 최적화한 라이브러리입니다.

- 학습 속도: **2배 향상**
- 메모리 사용량: **70% 절감**
- HuggingFace `transformers`와 완전 호환

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True,   # QLoRA
)

# LoRA 어댑터 추가
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 핵심 최적화
)
```

`use_gradient_checkpointing="unsloth"`가 Unsloth의 핵심 최적화입니다. 일반 gradient checkpointing보다 훨씬 효율적으로 메모리를 관리합니다.

---

## 5. Chat Template과 Label Masking

파인튜닝할 때 중요한 두 가지 개념입니다.

### 5-1. Chat Template

LLM에게 대화 형식으로 데이터를 먹이려면 모델이 기대하는 **정해진 포맷**에 맞춰야 합니다.

```python
# Gemma 형식 예시
"""
<start_of_turn>user
질문 내용<end_of_turn>
<start_of_turn>model
답변 내용<end_of_turn>
"""

# 코드로 적용
formatted = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
```

모델마다 포맷이 다르므로 반드시 해당 모델의 chat template을 확인해야 합니다.

### 5-2. Label Masking

학습 데이터에서 **User 발화 부분은 학습하지 않고, AI 응답 부분만 학습**하도록 마스킹합니다.

```python
# User 입력 토큰의 label을 -100으로 설정 → loss 계산에서 제외
labels = input_ids.clone()
labels[:, :user_end_idx] = -100  # User 발화 무시
```

LLM이 "어떻게 대답할지"만 학습하고, "어떻게 질문할지"는 학습하지 않도록 합니다.

---

## 6. 실습 예제 — 제주도 사투리 번역기

강의에서 진행한 파인튜닝 실습입니다. Qwen3 모델을 제주도 사투리 번역기로 파인튜닝했습니다.

```python
# 학습 데이터 예시
data = [
    {
        "instruction": "표준어를 제주도 사투리로 번역해줘",
        "input": "밥 먹었어?",
        "output": "밥 먹언?"
    },
    {
        "instruction": "표준어를 제주도 사투리로 번역해줘",
        "input": "어디 가?",
        "output": "어디 감수?"
    }
]

# SFTTrainer로 파인튜닝
from trl import SFTTrainer, TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        output_dir="./outputs"
    )
)
trainer.train()
```

불과 몇 백 개의 예시 데이터로, 소비자 GPU에서 수 시간 안에 특정 말투를 학습시킬 수 있었습니다.

---

## 7. 마무리

### 핵심 개념 요약

| 개념 | 설명 | 포인트 |
|------|------|--------|
| **Full Fine-tuning** | 전 파라미터 학습 | 성능 최고, 비용 최고 |
| **PEFT** | 소수 파라미터만 학습 | 효율과 성능의 균형 |
| **LoRA** | ΔW = A × B 저랭크 근사 | 파라미터 수십~백 배 절감 |
| **rank (r)** | LoRA 표현력 조절 | 보통 8~64 |
| **QLoRA** | 4-bit 양자화 + LoRA | 메모리 극단 절감 |
| **Unsloth** | LoRA 학습 최적화 | 2배 속도, 70% 메모리 절감 |
| **Chat Template** | 모델별 대화 형식 포맷 | 모델마다 다름, 반드시 확인 |
| **Label Masking** | AI 응답 부분만 학습 | User 입력은 -100 처리 |

day12의 핵심은 **적은 비용으로 LLM을 나만의 전문가로 만드는 방법**입니다. Full Fine-tuning이 아니어도, LoRA + QLoRA + Unsloth 조합으로 소비자 GPU에서도 충분히 실용적인 파인튜닝이 가능합니다.

---

**다음 Day 예고: Day 14 - 모델 양자화 + llama.cpp**

파인튜닝된 모델을 실제 서비스에 올리려면 추론 속도와 메모리 효율이 중요합니다. day14에서는 GGUF 포맷과 llama.cpp를 이용해 CPU에서도 빠르게 추론하는 방법, 그리고 TPS(Tokens Per Second)로 성능을 측정하는 방법을 다룹니다.

---

## 관련 포스트 추천
- Day 11 - Multi-turn + AI Agent 구현
- Day 14 - 모델 양자화 + llama.cpp (예정)
