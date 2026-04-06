---
title: [AI 입문] Python for AI — 꼭 알아야 할 핵심 정리 (NumPy · Pandas 포함)
tags: python, numpy, pandas, AI, 머신러닝, 공부기록
date: 2026-04-06
---

> **메타 디스크립션**: AI 입문 강의 day1 공부 기록. Python 기초 복습부터 NumPy 행렬 연산, Pandas 데이터 처리까지 — AI 개발에 꼭 필요한 핵심만 정리했습니다.
> **키워드**: Python AI, NumPy 행렬, Pandas 데이터프레임
> **예상 읽기 시간**: 6분
> **카테고리**: AI 입문 / Python
> **태그**: python, numpy, pandas, AI, 머신러닝, 공부기록

---

## 0. 들어가며

AI 강의 day1 내용을 정리한 공부 기록입니다.

Python 문법 자체는 어느 정도 알고 있는데, AI/데이터 관점에서 어떤 부분이 중요한지 감이 잘 안 잡혔습니다. 이번 강의에서 "Python 기초 → NumPy → Pandas" 순서로 흐름을 잡아줘서 정리해두면 나중에 유용할 것 같아 글로 남깁니다.

대상 독자는 저처럼 **Python 기초는 있지만 AI/데이터 쪽으로 방향을 잡고 싶은 개발자나 취준생**입니다. 개념 설명보다는 코드 중심으로 빠르게 훑어볼 수 있도록 구성했습니다.

---

## 1. Python 기초 복습

### 변수와 타입 힌트

AI 코드에서는 타입 힌트가 자주 등장합니다. 읽는 사람 입장에서 "이 변수가 어떤 타입인지" 바로 알 수 있어서 협업할 때 특히 유용합니다.

```python
a = 10
c: int = 55  # 타입 힌트 (Python 3.11+ 권장)
```

### if / for 제어문

기본 중의 기본이지만, `end=''` 같은 출력 옵션은 가끔 헷갈리니 같이 정리해둡니다.

```python
if c == 55:
    print('HUHU')

for i in range(10):
    print(i, end='')  # 줄바꿈 없이 출력
```

### 함수 정의 — 타입 힌트 3단계

타입 힌트를 얼마나 붙이느냐에 따라 3단계로 나뉩니다. AI 프레임워크 소스코드를 보면 `sum3` 스타일이 압도적으로 많습니다.

```python
# 단계 1: 타입 힌트 없음
def sum1(a, b):
    return a + b

# 단계 2: 인자에만 타입 힌트
def sum2(a: int, b: int):
    return a + b

# 단계 3: 인자 + 반환값 모두 타입 힌트
def sum3(a: int, b: int) -> int:
    return a + b
```

### List 인덱싱

Python 리스트는 다양한 타입을 섞어도 됩니다. 인덱스는 0부터 시작한다는 것, 기억하고 계시죠?

```python
lst = [1, 2, 'HI', 4, 5]
print(lst[0])  # 1
```

### Class — instance attribute와 method

클래스에서 `__init__`은 인스턴스를 만들 때 자동 실행됩니다. `self.name`처럼 `self`에 붙어있는 변수가 인스턴스 속성(instance attribute)입니다.

```python
class Person:
    def __init__(self):
        self.name = 'HI'   # instance attribute
        self.age = '30'

    def sayHi(self):
        print('안녕')

    def sayBye(self):
        print(f'잘가 {self.name} 님')
```

---

## 2. NumPy — AI를 위한 수치 계산

### 왜 NumPy인가?

AI에서 다루는 데이터는 결국 숫자의 배열입니다. 이미지 한 장도, 텍스트도, 결국 숫자 행렬로 변환됩니다. NumPy는 이런 대규모 수치 계산을 Python 순수 반복문보다 수십 배 빠르게 처리해 주는 라이브러리입니다.

> 💡 **핵심 포인트**: NumPy 없이 AI 코드를 읽기 어렵습니다. 배열 연산의 기본기가 곧 AI 코드 독해력입니다.

### 1D 벡터 vs 2D 행렬

```python
import numpy as np

# 벡터 (1D)
v = np.array([3, 4])
print(np.linalg.norm(v))  # 크기(norm) = 5.0

# 행렬 (2D)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
```

### 배열 초기화

새 배열을 만들 때 자주 쓰이는 패턴 4가지입니다.

```python
np.zeros((3, 5))          # 0으로 채운 3x5 행렬
np.ones((2, 2))           # 1로 채운 2x2 행렬

np.random.seed(42)
np.random.rand(2, 3)      # 0~1 사이 랜덤 값으로 채운 2x3 행렬

np.eye(3)                 # 3x3 단위행렬 (대각선만 1)
```

`random.seed(42)`를 쓰면 실행할 때마다 같은 랜덤 값이 나옵니다. 실험 재현성을 위해 습관적으로 넣어두는 게 좋습니다.

### 요소별 연산 & 수학 함수

```python
A = np.array([4, 16, 25])

np.sqrt(A)   # [2.  4.  5.]  — 제곱근
np.log(A)    # 자연로그 (밑수 e)
np.exp(A)    # e의 지수함수
```

Python 기본 `math.sqrt()`는 스칼라 하나에만 쓸 수 있지만, NumPy 함수는 배열 전체에 한 번에 적용됩니다. 이걸 **브로드캐스팅(broadcasting)** 이라고 부릅니다.

### 선형변환: 스칼라 배와 행렬 곱 (@)

```python
c = 3
v = np.array([[1, 2, 2],
              [1, 1, 5]])

# 스칼라 배 — 모든 원소에 3을 곱함
result = c * v

# 행렬 곱 (@) — 선형변환
S = np.array([[2, 0],
              [0, 1]])   # x축 방향으로 2배 늘리는 변환행렬

v_scaled = S @ v
```

`@` 연산자는 Python 3.5+에서 행렬 곱을 의미합니다. `np.dot()`와 같은 결과지만 코드가 훨씬 읽기 편합니다.

### 배열 조작: transpose (.T) & triu

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A.T          # 전치 — (2,3) → (3,2)
np.triu(A)   # 상삼각행렬 (대각선 아래를 0으로)
```

### ndarray 속성 한눈에 보기

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.size)   # 6      — 전체 원소 수
print(A.ndim)   # 2      — 차원 수
print(A.shape)  # (2, 3) — 행 x 열
```

`shape`를 제일 많이 씁니다. 에러 디버깅할 때 shape 불일치가 원인인 경우가 절반은 되는 것 같습니다.

---

## 3. Pandas — 데이터 다루기 첫걸음

### DataFrame이란?

엑셀 시트를 코드로 다룬다고 생각하면 됩니다. 행(row)과 열(column)로 구성된 2차원 테이블 구조로, 실제 AI 프로젝트에서 데이터 전처리의 대부분은 Pandas로 이루어집니다.

### Iris 데이터셋으로 보는 기본 조작

실습에서 가장 많이 쓰이는 예제 데이터셋인 Iris(붓꽃)를 활용합니다.

```python
import pandas as pd
import seaborn as sns

# 데이터 로드
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
df.head(10)  # 상위 10행 확인
```

**불필요한 열 제거 (drop)**

```python
# sepal(꽃받침) 관련 열은 제거하고 petal(꽃잎)만 남기기
df_sub = df.drop(columns=['sepal_length', 'sepal_width'])
```

**조건 필터링**

```python
# petal_width가 0.3 이상인 행만 필터링
df_filtered = df_sub[df_sub["petal_width"] >= 0.3]
```

SQL의 `WHERE` 조건과 같은 개념입니다. 대괄호 안에 조건식을 그대로 넣으면 됩니다.

### Seaborn으로 산점도 시각화

```python
sns.scatterplot(
    data=df_filtered,
    x="petal_length",
    y="petal_width",
    hue="species"    # 품종별로 색상 구분
)
```

`hue` 파라미터 하나로 그룹별 색상 구분이 끝납니다. Matplotlib보다 코드가 훨씬 간결해서 탐색적 데이터 분석(EDA)에 자주 쓰입니다.

[이미지 위치: Iris 데이터셋 산점도 — petal_length vs petal_width, 품종별 색상 구분 — alt: "붓꽃 데이터셋 꽃잎 길이 너비 산점도"]

---

## 4. 마무리

day1에서 다룬 내용을 한 줄로 요약하면, **"AI 코드를 읽고 쓰기 위한 도구 삼총사 세팅"** 입니다.

### 핵심 요약

| 라이브러리 | 주요 용도 | 핵심 함수/속성 |
|---|---|---|
| Python | 로직 구현, 클래스 설계 | 타입 힌트, `def`, `class` |
| NumPy | 수치 계산, 행렬 연산 | `np.array`, `@`, `.T`, `.shape` |
| Pandas | 테이블 데이터 전처리 | `head()`, `drop()`, 조건 필터링 |

### 다음 학습 방향

day2는 **AI Math** 파트로 넘어갈 예정입니다. 선형대수(행렬, 벡터), 확률/통계 기초가 주요 내용인 것 같습니다. NumPy에서 맛보기로 접했던 행렬 곱이 본격적으로 등장할 것 같아서 기대됩니다.

이 글이 AI 입문을 준비 중인 분들께 조금이라도 도움이 됐으면 좋겠습니다. 틀린 내용이 있으면 댓글로 알려주세요!

---

## 관련 포스트 추천
- [AI 입문] AI Math — 선형대수와 확률 핵심 정리 (day2)
- Python 타입 힌트 완전 정복 — 왜 AI 코드에서 필수인가
- NumPy 브로드캐스팅 완전 이해 — 배열 연산의 마법
- Pandas 데이터 전처리 실전 패턴 10가지

---

#python #numpy #pandas #AI #머신러닝 #공부기록
