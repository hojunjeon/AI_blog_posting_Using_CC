---
title: "Multi-turn 대화와 AI Agent — LLM이 스스로 생각하고 행동하다"
tags: [AIAgent, MultiTurn, ReAct, LangChain, LangGraph, InMemorySaver, Gradio, ToolCalling]
date: 2026-04-14
---

> **메타 디스크립션**: AI 입문 강의 day11 공부 기록. 대화 히스토리 유지하는 Multi-turn 구현, 토큰 비용 줄이는 요약 전략, 그리고 ReAct 패턴 기반 AI Agent를 LangChain/LangGraph로 구현하는 과정을 정리했습니다.
> **키워드**: AI Agent, Multi-turn, ReAct, Tool Calling, LangChain, LangGraph, Gradio
> **예상 읽기 시간**: 12분
> **카테고리**: AI 입문 / LLM 응용
> **태그**: AIAgent, MultiTurn, ReAct, LangChain, LangGraph, Gradio

---

## 0. 들어가며

AI 강의 day11 내용을 정리한 공부 기록입니다.

day10에서 RAG와 LangChain 기초를 다뤘다면, day11은 한 단계 더 나아갑니다. 대화 히스토리를 유지하는 **Multi-turn**, LLM이 스스로 도구를 선택하고 사용하는 **AI Agent**, 그리고 이 모든 걸 UI로 연결하는 **Gradio**까지입니다.

이번 강의의 흐름:

```
Multi-turn 대화 구현
  → 토큰 비용 관리 (요약)
  → AI Agent (ReAct 패턴)
  → Agent + RAG 결합
  → Gradio UI 연결
```

---

## 1. Multi-turn — 대화 히스토리 유지하기

기본 LLM 호출은 매번 독립적입니다. "이전에 뭐라고 했지?" 같은 맥락을 모릅니다. Multi-turn은 대화 히스토리를 직접 관리해서 LLM이 이전 맥락을 참조하도록 합니다.

### 1-1. History 리스트로 관리

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

history = []

def chat(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    
    return response.content
```

`history` 리스트에 `HumanMessage`와 `AIMessage`를 번갈아 쌓으면, LLM이 이전 대화 전체를 보면서 답변합니다. 직관적이지만 문제가 하나 있습니다.

### 1-2. 토큰 비용 문제

대화가 길어질수록 history가 누적됩니다. LLM API는 입력 토큰 길이에 비례해서 비용이 증가하므로, 대화가 100턴을 넘으면 초반 대화까지 모두 포함된 긴 입력이 매번 전송됩니다.

**해결책: 요약(Summarization)**

```python
def summarize_history(history: list) -> str:
    summary_prompt = [
        SystemMessage(content="다음 대화를 요약해줘. 핵심 정보만 남겨."),
        *history
    ]
    summary = model.invoke(summary_prompt)
    return summary.content

# 대화가 10턴을 넘으면 요약
if len(history) > 10:
    summary = summarize_history(history[:-2])
    history = [SystemMessage(content=f"이전 대화 요약: {summary}")] + history[-2:]
```

오래된 대화는 요약으로 압축하고 최근 몇 턴만 원본으로 유지합니다. 맥락은 유지하면서 토큰 비용을 줄일 수 있습니다.

---

## 2. LangChain v1.0 — 무엇이 달라졌나

LangChain은 v0에서 v1.0으로 넘어오면서 큰 변화가 있었습니다.

| 버전 | 방식 | 특징 |
|------|------|------|
| v0 | Chain 중심 | `ConversationChain`, `LLMChain` 등 미리 만들어진 체인 |
| v1.0+ | Agent/LCEL 중심 | `|` 연산자로 자유롭게 파이프라인 구성 |

v0 방식은 사용하기 편했지만 유연성이 부족했습니다. v1.0부터는 LCEL(LangChain Expression Language)로 컴포넌트를 직접 연결하는 방식이 표준이 되었습니다.

---

## 3. AI Agent — LLM이 스스로 판단하고 행동하다

AI Agent의 핵심은 **LLM이 어떤 도구를 언제 사용할지 스스로 결정한다**는 점입니다.

### 3-1. Tool 정의

```python
from langchain_core.tools import tool

@tool
def add_sum(a: int, b: int) -> int:
    """두 수를 더하는 도구"""
    return a + b

@tool
def add_minus(a: int, b: int) -> int:
    """두 수를 빼는 도구"""
    return a - b

tools = [add_sum, add_minus]
```

도구 함수의 **docstring**이 중요합니다. LLM이 docstring을 읽고 어떤 도구를 써야 할지 판단하기 때문입니다.

### 3-2. ReAct 패턴 — 생각하고, 행동하고, 관찰하고

ReAct(Reasoning + Acting)는 AI Agent의 동작 방식입니다.

```
[질문]: "3 + 5는 얼마야?"

Thought: 덧셈이 필요하다. add_sum 도구를 써야겠다.
Action: add_sum(a=3, b=5) 호출
Observation: 결과 = 8
Final Answer: 3 + 5 = 8 입니다.
```

단순히 LLM이 답변하는 게 아니라, **생각 → 행동 → 관찰 → 다시 생각** 루프를 돌며 복잡한 문제를 단계적으로 해결합니다.

### 3-3. LangGraph로 Agent 구현

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# 메모리 설정 (대화 히스토리 자동 관리)
checkpointer = InMemorySaver()

# ReAct Agent 생성
agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer
)

# 실행
config = {"configurable": {"thread_id": "session_1"}}
result = agent.invoke(
    {"messages": [HumanMessage(content="3 더하기 5는?")]},
    config=config
)
```

`create_react_agent`는 ReAct 루프를 자동으로 처리합니다. `InMemorySaver`는 `thread_id` 단위로 대화 히스토리를 자동 관리해줍니다.

### 3-4. Agent + RAG 결합

도구 목록에 RAG 검색 함수를 넣으면 Agent가 필요할 때 자동으로 문서 검색을 수행합니다.

```python
@tool
def search_docs(query: str) -> str:
    """사내 문서에서 관련 내용을 검색합니다"""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs[:3]])

# RAG 검색도 도구 목록에 포함
tools = [add_sum, add_minus, search_docs]

agent = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)
```

이제 수학 계산이 필요하면 `add_sum`을, 문서 검색이 필요하면 `search_docs`를 스스로 선택합니다.

---

## 4. Gradio — 웹 UI로 연결하기

Gradio는 파이썬 코드 몇 줄로 웹 인터페이스를 만들어주는 라이브러리입니다.

```python
import gradio as gr

def respond(user_message, chat_history):
    config = {"configurable": {"thread_id": "gradio_session"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )
    bot_message = result["messages"][-1].content
    chat_history.append((user_message, bot_message))
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="질문을 입력하세요")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
```

`demo.launch()`를 실행하면 로컬 주소로 웹 채팅 UI가 열립니다. AI Agent와 실시간으로 대화할 수 있는 인터페이스가 완성됩니다.

---

## 5. 마무리

### 핵심 개념 요약

| 개념 | 설명 | 포인트 |
|------|------|--------|
| **Multi-turn** | 대화 히스토리 유지 | History 리스트에 메시지 누적 |
| **요약 전략** | 오래된 대화 압축 | 토큰 비용 절감 |
| **ReAct 패턴** | 생각→행동→관찰 루프 | 복잡한 문제 단계적 해결 |
| **AI Agent** | LLM이 도구를 스스로 선택 | docstring이 도구 선택 기준 |
| **InMemorySaver** | 세션별 히스토리 자동 관리 | `thread_id`로 세션 구분 |
| **Agent + RAG** | 검색 도구를 Agent에 통합 | 필요 시 자동 문서 검색 |
| **Gradio** | Python으로 웹 UI 생성 | `demo.launch()`로 즉시 실행 |

day11의 핵심은 LLM이 **수동적인 응답기에서 능동적인 에이전트로** 진화한다는 개념입니다. 사람이 어떤 도구를 쓸지 지정하는 게 아니라, LLM이 상황을 판단해서 스스로 행동합니다. ReAct 패턴을 이해하고 나면 AI Agent가 왜 강력한지 실감하게 됩니다.

---

**다음 Day 예고: Day 12 - PEFT · LoRA · QLoRA**

LLM을 내 목적에 맞게 효율적으로 파인튜닝하는 방법을 다룹니다. 전체 파라미터를 건드리지 않고 소수의 파라미터만 업데이트하는 LoRA, 메모리를 극단적으로 줄이는 QLoRA, 그리고 Unsloth로 2배 빠르게 학습하는 방법까지입니다.

---

## 관련 포스트 추천
- Day 10 - LangChain + RAG: 문서 검색 기반 AI 만들기
- Day 12 - PEFT · LoRA · QLoRA: 효율적인 파인튜닝 (예정)
