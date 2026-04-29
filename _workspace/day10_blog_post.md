---
title: "LangChain + RAG — LLM에 문서 검색 능력을 더하다"
tags: [LangChain, RAG, ChromaDB, LangGraph, 임베딩, VectorDB, Retriever, LangSmith]
date: 2026-04-09
---

> **메타 디스크립션**: AI 입문 강의 day10 공부 기록. LangChain으로 파이프라인 구성하는 법, Tool/Agent 개념, 그리고 RAG(Retrieval-Augmented Generation)를 LangGraph로 구현하는 과정까지 정리했습니다.
> **키워드**: LangChain, RAG, ChromaDB, 임베딩, LangGraph, Retriever, VectorDB
> **예상 읽기 시간**: 12분
> **카테고리**: AI 입문 / LLM 응용
> **태그**: LangChain, RAG, ChromaDB, LangGraph, 임베딩

---

## 0. 들어가며

AI 강의 day10 내용을 정리한 공부 기록입니다.

day8까지는 LLM에 프롬프트를 넣고 결과를 받는 수준이었다면, day10부터는 본격적으로 **LLM 애플리케이션을 만드는 방법**을 다룹니다. LangChain이라는 프레임워크를 통해 파이프라인을 연결하고, RAG 기술로 LLM에 외부 문서 검색 능력을 더하는 흐름입니다.

이번 강의의 큰 그림은 이렇습니다.

```
[LangChain 기초]
  프롬프트 → LLM → 파서 파이프라인 구성

[Tool / Agent]
  LLM이 스스로 도구를 선택해서 호출하도록

[RAG]
  PDF → Chunk → 임베딩 → VectorDB → 검색 → LLM 답변
```

---

## 1. LangChain — LLM 파이프라인 프레임워크

LangChain은 LLM 작업 흐름을 연결해주는 프레임워크입니다. 단순히 LLM을 호출하는 것이 아니라 다음 과정을 코드로 쉽게 구성할 수 있습니다.

```
입력 → 프롬프트 가공 → LLM 호출 → 출력 가공 → 다음 단계
```

### 1-1. 기본 체인 구성

LangChain의 핵심은 `|` 연산자로 컴포넌트를 이어붙이는 것입니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 모델 정의 (Upstage solar-pro 사용)
model = ChatOpenAI(
    model="solar-pro",
    base_url="https://api.upstage.ai/v1",
    api_key=API_KEY
)

# 프롬프트 템플릿: {question} 자리에 실제 질문이 들어감
prompt = ChatPromptTemplate.from_template(
    "다음 질문에 답해줘: {question}"
)

# 체인 구성: 프롬프트 | 모델 | 파서
chain = prompt | model | StrOutputParser()

# 호출
result = chain.invoke({"question": "치킨은 왜 맛있어?"})
```

`prompt | model | StrOutputParser()` 이 한 줄이 LangChain의 핵심입니다. 각 컴포넌트의 출력이 다음 컴포넌트의 입력으로 자동 연결됩니다.

---

## 2. Tool — LLM이 스스로 선택해서 호출하는 도구

LLM이 내부적으로 모르는 정보(최신 데이터, 내부 시스템 정보 등)를 필요로 할 때 **Tool**을 사용합니다.

```
LLM이 스스로 판단 → "이 질문은 검색이 필요하다" → Tool 호출 → 결과를 답변에 반영
```

### 2-1. Tool 정의

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """내부 정보를 검색하는 도구"""
    fake_db = {
        "김지윤": "완전 공주고, 공부도 잘하고, 성격도 좋고, 멋진 사람이다.",
        "달러환율": "1511원입니다"
    }
    for key in fake_db:
        if key in query:
            return fake_db[key]
    return "검색 결과 없음"

tools = [search_tool]
```

`@tool` 데코레이터 하나로 일반 Python 함수를 LLM이 호출 가능한 도구로 만들 수 있습니다.

### 2-2. 모델에 Tool 바인딩

```python
# Tool을 모델에 연결
model_with_tools = model.bind_tools(tools)

# 체인 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "인물 정보나 환율 정보가 필요하면 반드시 search_tool을 사용해야 한다."),
    ("human", "{question}")
])

chain = prompt | model_with_tools
```

`model.bind_tools(tools)`로 모델에 도구 목록을 알려주면, LLM이 응답할 때 도구 호출이 필요한지 스스로 판단합니다.

### 2-3. Tool 실행 흐름

```python
response = chain.invoke({"question": "김지윤이 누구야?"})

# LLM이 tool_calls를 반환했는지 확인
if response.tool_calls:
    for tool_call in response.tool_calls:
        result = search_tool.invoke(tool_call["args"])

        final_response = model.invoke([
            ("system", "검색 결과를 기반으로 대답해"),
            ("human", f"질문: {query} / 검색 결과: {result}")
        ])
```

LLM이 `tool_calls`를 반환하면 직접 Tool을 실행하고, 그 결과를 다시 LLM에 넣어 최종 답변을 얻습니다.

---

## 3. RAG — LLM에 문서 검색 능력을 더하다

### 3-1. RAG가 필요한 이유

LLM은 학습 데이터에 없는 정보를 모릅니다. 사내 문서, 최신 데이터, 비공개 정보 같은 건 GPT도 모릅니다. RAG는 이 문제를 해결하는 기술입니다.

```
프롬프트만 넣기: "오늘 날씨 어때?"  →  LLM: 모름 (학습 데이터에 없음)
RAG 적용:       "오늘 날씨 어때? (검색 결과: 비옴)"  →  LLM: "오늘은 비가 와요"
```

질문을 LLM에 넣기 전, **관련 내용을 검색해서 프롬프트를 보강**하는 기술이 RAG입니다.

### 3-2. RAG 준비 단계

RAG를 구현하려면 다음 준비물이 필요합니다.

| 준비물 | 역할 |
|--------|------|
| 문서 텍스트 | 검색 대상이 되는 원본 데이터 |
| 임베딩 모델 | 텍스트를 의미 벡터로 변환 |
| Vector DB | 벡터를 저장하고 검색 |
| Retriever | 질문과 유사한 Chunk를 찾아줌 |

이번 실습에서는 Upstage 임베딩 모델과 ChromaDB를 사용합니다.

### 3-3. Chunk — 문서를 조각내는 이유

LLM은 한 번에 처리할 수 있는 입력 길이에 한계가 있습니다. 그래서 큰 문서를 **Chunk**라는 단락 단위로 나눠서 저장합니다.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF 로드
loader = PyPDFLoader("./chayoon_lego/chayoon_lego_world.pdf")
documents = loader.load()

# 200자 단위로 자르기 (40자 겹치기)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)
chunks = splitter.split_documents(documents)
```

`chunk_overlap=40`은 정보가 Chunk 경계에서 잘리는 것을 방지하기 위해 앞뒤 40자를 겹치게 저장하는 설정입니다.

> Chunk 크기는 보통 300~1000 토큰(한글 약 20문장) 정도가 실용적입니다.

### 3-4. ChromaDB에 임베딩 저장

```python
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

# Upstage 임베딩 모델
embeddings = UpstageEmbeddings(model="embedding-query")

# Chunk들을 임베딩해서 VectorDB에 저장
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_store",  # 로컬 파일로 저장
)
vector_store.add_documents(chunks)
```

`persist_directory`를 지정하면 프로그램 종료 후에도 데이터가 유지됩니다.

### 3-5. Retriever 테스트

```python
retriever = vector_store.as_retriever()

question = "차윤이 손 모양은?"
retrieved_docs = retriever.invoke(question)

print(retrieved_docs[0].page_content)
```

질문을 임베딩 벡터로 변환하고 Cosine 유사도로 가장 관련 높은 Chunk를 찾아줍니다. 완벽히 같은 단어가 아니어도 **의미가 유사한** 내용을 검색할 수 있습니다.

---

## 4. LangGraph로 RAG 워크플로우 구현

RAG 파이프라인을 LangGraph의 그래프 구조로 구현하면 각 단계를 노드로 관리할 수 있습니다.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# 1. State 정의
class State(TypedDict):
    messages: Annotated[list, operator.add]

# 2. RAG 노드 함수 정의
def rag_node(state: State):
    question = state["messages"][-1].content

    # 관련 문서 검색
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    # LLM 호출 (질문 + 검색 결과)
    response = model.invoke([
        SystemMessage(content=f"다음 정보를 참고해서 대답해:\n{context}"),
        HumanMessage(content=question)
    ])
    return {"messages": [response]}

# 3. 그래프 연결
graph = StateGraph(State)
graph.add_node("rag", rag_node)
graph.add_edge(START, "rag")
graph.add_edge("rag", END)
app = graph.compile()
```

LangGraph를 쓰면 RAG 흐름 외에도 조건 분기, 루프, 멀티 에이전트 구성이 가능해집니다.

---

## 5. LangSmith — 내부 흐름 디버깅

LangSmith는 LangChain 기반 앱의 트레이싱 도구입니다. 어떤 Chunk가 검색되었는지, LLM에 어떤 프롬프트가 들어갔는지 시각적으로 확인할 수 있습니다.

RAG가 엉뚱한 답변을 반환할 때 원인을 추적하는 데 특히 유용합니다.

---

## 6. 마무리

### 핵심 개념 요약

| 개념 | 설명 | 포인트 |
|------|------|--------|
| **LangChain** | LLM 파이프라인 프레임워크 | `prompt \| model \| parser` 체인 구성 |
| **Tool** | LLM이 필요할 때 호출하는 도구 | `@tool` + `bind_tools()` |
| **RAG** | 검색 결과를 LLM에 주입하는 기술 | 비공개 정보, 최신 데이터에 유효 |
| **Chunk** | 문서를 나눈 단락 단위 | 300~1000 토큰, overlap 설정 |
| **임베딩** | 텍스트를 의미 벡터로 변환 | 유사 의미 검색 가능 |
| **ChromaDB** | 벡터 저장 + 검색 DB | `persist_directory`로 로컬 저장 |
| **Retriever** | 유사 Chunk 검색기 | `vector_store.as_retriever()` |
| **LangGraph** | 워크플로우 그래프 프레임워크 | 노드/엣지로 흐름 관리 |
| **LangSmith** | 실행 트레이싱 도구 | 디버깅, 검색 결과 확인 |

day10의 핵심은 **LLM 혼자서는 모르는 정보를 외부 검색으로 보완한다**는 RAG 패러다임입니다. 단순 프롬프트에서 벗어나 실제 문서 기반 AI 시스템을 구축하는 첫 단계였습니다.

---

**다음 Day 예고: Day 11 - Multi-turn + AI Agent**

대화 히스토리를 유지하는 Multi-turn 구현, 토큰 비용을 줄이는 요약 기법, 그리고 ReAct 패턴 기반 AI Agent를 다룰 예정입니다.

---

## 관련 포스트 추천
- Day 8 - LLM으로 학습 데이터 만들기
- Day 7 - 토크나이저부터 임베딩까지
- Day 11 - Multi-turn + AI Agent 구현 (예정)
