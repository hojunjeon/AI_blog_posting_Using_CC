---
title: [AI 입문] EDA + Scikit-learn — 데이터 탐색부터 ML 파이프라인까지
tags: EDA, Scikit-learn, 데이터전처리, 표준화, 머신러닝파이프라인, 공부기록
date: 2026-04-07
---

> **메타 디스크립션**: AI 입문 강의 day3 공부 기록. EDA로 데이터를 탐색하고 Scikit-learn으로 ML 파이프라인을 구성하는 과정을 정리했습니다.
> **키워드**: EDA 탐색적 데이터 분석, Scikit-learn 파이프라인, 표준화 정규화 차이, train test split
> **예상 읽기 시간**: 8분
> **카테고리**: AI 입문 / EDA & Scikit-learn
> **태그**: EDA, Scikit-learn, 데이터전처리, 표준화, 머신러닝파이프라인, 공부기록

---

## 0. 들어가며

AI 강의 day3 내용을 정리한 공부 기록입니다.

day2까지는 NumPy로 선형회귀·로지스틱회귀를 직접 구현했는데, day3부터는 실전에 가까운 흐름으로 바뀝니다. 데이터를 받았을 때 어떻게 살펴보고, 어떻게 전처리하고, 어떻게 Scikit-learn에 태우는지 — ML 파이프라인 전체의 흐름을 다루는 파트입니다.

솔직히 EDA는 "그냥 데이터 보는 거 아닌가?" 싶었는데, 배우고 나서 보니 상관관계 분석이나 분포 확인 같은 작업이 전처리 방향을 결정하는 핵심이라는 게 이해가 됐습니다. 이번 글도 코드 중심으로 정리해 둡니다.

---

## 1. ML 파이프라인 전체 흐름

본격적인 내용 전에 전체 그림을 먼저 잡아두면 좋습니다.

```
1. 데이터 확인       — 크기, 구조, 원하는 컬럼 있는지
2. 데이터 분할       — Train / Validation / Test
3. EDA               — 분포, 이상치, 결측치, 상관관계 파악
4. 전처리            — 결측치 처리, 스케일링, 인코딩
5. 모델 학습         — 파라미터 확정
6. 평가              — 필요 시 3~5 반복
```

### 데이터 분할 비율

| 구분 | 비율 | 역할 |
|---|---|---|
| **Train** | 60~80% | 파라미터 학습 |
| **Validation** | 10~20% | 하이퍼파라미터 튜닝, 과적합 탐지 |
| **Test** | 10~20% | 최종 성능 평가 — **딱 한 번만** 사용 |

테스트 셋은 절대로 학습 도중 들여다보면 안 됩니다. 학교에서 시험 답지를 미리 보고 공부하는 것과 같아서, 모델이 새 데이터에 얼마나 잘 일반화되는지 측정이 불가능해집니다. Scikit-learn의 `train_test_split`으로 간단하게 나눌 수 있습니다.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 2. EDA (탐색적 데이터 분석)

EDA는 **데이터를 본격적으로 건드리기 전에 탐색하는 행위** 전체를 가리킵니다. csv 파일 받고 `df.head()` 찍어보는 것도 EDA고, 이상치를 시각화해서 확인하는 것도 EDA입니다.

강의에서는 seaborn 내장 `mpg` 데이터셋(차량 연비 데이터)으로 실습했습니다.

```python
import seaborn as sns
df = sns.load_dataset("mpg")
```

### 기초 탐색 3종 세트

```python
df.head()       # 상위 5행 미리보기
df.info()       # 컬럼별 타입, Non-Null 개수 → 결측치 파악
df.describe()   # count, mean, std, min, 25%, 50%, 75%, max
```

`df.info()`에서 `Non-Null Count`가 전체 행 수보다 작으면 결측치가 있다는 뜻입니다. mpg 데이터에서는 `horsepower` 컬럼에 결측치가 6개 있습니다.

`df.describe()`의 `std`(표준편차)는 데이터가 평균 주변에 얼마나 퍼져 있는지를 보여줍니다. 표준편차가 크면 값의 분포가 넓고, 작으면 평균에 집중되어 있습니다. AI 학습에서 표준편차는 표준화와 바로 연결되기 때문에 중요한 값입니다.

> 💡 **핵심 포인트**: EDA는 데이터를 수정하거나 처리하는 것이 아닙니다. 이상치와 결측치를 "확인"하는 것은 EDA지만, 그것을 "삭제하거나 채우는 것"은 전처리입니다.

---

## 3. 상관관계 분석

EDA에서 가장 많이 쓰는 분석 중 하나가 상관관계입니다. **두 변수가 함께 변하는 경향이 있는지** 수치로 파악할 수 있습니다.

### 산점도 + 회귀선

```python
import seaborn as sns

sns.scatterplot(data=df, x="weight", y="acceleration", alpha=0.7)
sns.regplot(data=df, x="weight", y="acceleration", scatter_kws={'alpha':0.6})
```

시각화를 보면 차량 무게(`weight`)가 무거울수록 가속(`acceleration`, 100m 도달 시간)이 느리다는 경향이 보입니다. 이 경향의 강도를 수치로 표현하면 상관계수입니다.

### 상관계수 해석

```python
corr = df["weight"].corr(df["acceleration"])
```

| 값 | 의미 |
|---|---|
| +1 | 완전 양의 상관 (한쪽 오르면 다른 쪽도 오름) |
| +0.7 이상 | 강한 양의 상관 |
| 0 | 무상관 |
| -0.7 이하 | 강한 음의 상관 |
| -1 | 완전 음의 상관 |

> **주의**: 상관관계 ≠ 인과관계입니다. "A와 B가 함께 변한다"는 것이지 "A가 B를 일으킨다"는 뜻이 아닙니다.

### 히트맵으로 전체 상관관계 한눈에 보기

```python
import numpy as np

corr = df.corr(numeric_only=True)

# 삼각형 마스크 (대칭 행렬의 중복 제거)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", linewidths=1)
```

히트맵으로 찍어보니 몇 가지 패턴이 보였습니다.

- `model_year`가 올라갈수록 연비(`mpg`)도 좋아지는 추세
- `cylinders`, `displacement`, `weight`, `horsepower`는 서로 강한 양의 상관 → 중복 변수일 가능성
- `acceleration`은 `weight`와 음의 상관 (무거울수록 가속이 느림)

---

## 4. 분포(Distribution)와 분산

### 히스토그램

```python
sns.histplot(df["weight"], bins=20, kde=True, color="skyblue")
```

`kde=True`를 주면 히스토그램 위에 확률 밀도 곡선(KDE)을 같이 그려줍니다. mpg 데이터에서 차량 무게는 2,000 이후에 가장 많이 분포하고 있었습니다.

### 분산과 표준편차

분포를 수치로 요약할 때 중요한 두 통계값입니다.

- **분산(Variance)**: 각 값이 평균에서 얼마나 떨어져 있는지의 제곱 평균
- **표준편차(Standard Deviation)**: 분산의 제곱근 → 원래 단위로 해석 가능

```python
import numpy as np

data_a = np.array([79, 80, 81])  # 분산 작음
data_b = np.array([70, 80, 90])  # 분산 큼

print(np.var(data_a))   # → 0.666...
print(np.var(data_b))   # → 66.666...
```

AI에서는 **평균과 표준편차** 이 두 값이 핵심입니다. 표준화가 바로 "평균을 0으로, 표준편차를 1로" 만드는 작업이기 때문입니다.

---

## 5. pairplot — 상관관계 한 방에 전부 보기

EDA할 때 변수가 여러 개면 모든 조합의 산점도를 하나씩 그리기 번거롭습니다. `pairplot`을 쓰면 한 번에 해결됩니다.

```python
df_clean = df.dropna(subset=["horsepower"])  # 결측치 제거

sns.pairplot(
    df_clean[["weight", "horsepower", "acceleration"]],
    diag_kind="kde"
)
```

대각선에는 각 변수의 분포(KDE), 나머지 칸에는 변수 간 산점도가 그려집니다. 강한 상관관계가 있으면 산점도가 대각선 방향으로 모이고, 무상관이면 퍼져 있습니다.

---

## 6. Scikit-learn 기본 패턴

EDA로 데이터를 이해했으면 이제 모델을 학습할 차례입니다. Scikit-learn은 머신러닝의 표준 라이브러리로, 거의 모든 모델이 **fit → predict** 구조를 따릅니다.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)       # 학습
y_pred = model.predict(X_test)   # 예측
```

day2에서 NumPy로 100줄 넘게 짰던 선형회귀가 이렇게 세 줄로 끝납니다. Scikit-learn은 **같은 인터페이스(fit/predict)** 로 모든 모델을 다루기 때문에, 모델을 바꿔도 코드 구조가 그대로입니다.

---

## 7. 표준화(Standardization)

### 왜 필요한가?

예를 들어 키(160~190 cm)와 발 길이(0.5~1.5 cm)를 같이 학습하면, 키 변수의 숫자가 훨씬 크기 때문에 경사하강법이 키 쪽으로 편향됩니다. 발 길이가 실제로는 중요한 변수여도 제대로 반영이 안 되는 것입니다.

표준화는 모든 변수를 같은 스케일로 맞춰서 이 문제를 해결합니다.

### 표준화 공식

$$z = \frac{x - \mu}{\sigma}$$

결과: **평균 0, 표준편차 1**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 학습 + 변환
X_test_scaled  = scaler.transform(X_test)       # 변환만 (fit 없이!)
```

### ⚠️ 테스트 데이터에 fit 금지 — 데이터 누수(Data Leakage)

이 부분이 꽤 중요한 포인트입니다. 테스트 데이터로 `fit_transform`을 쓰면 테스트 데이터의 통계값(평균, 표준편차)이 스케일러에 반영되고, 그 정보가 모델 평가에 새어 들어가 실제 성능보다 좋게 측정됩니다.

- 학습 데이터: `fit_transform()` — 통계값 계산 + 변환
- 테스트 데이터: `transform()` — 학습 때 구한 통계값으로 변환만

### 표준화 후 예측 — 역변환도 필요

```python
# 테스트
new_bill = [[50]]
new_bill_scaled = scaler_X.transform(new_bill)      # 표준화
pred_scaled     = model.predict(new_bill_scaled)    # 예측
pred_tip        = scaler_y.inverse_transform(pred_scaled)  # 역변환
```

표준화된 데이터로 학습했으면 예측값도 표준화된 공간에 있습니다. 실제 단위로 해석하려면 `inverse_transform`으로 되돌려야 합니다.

> 💡 **핵심 포인트**: 표준화(StandardScaler)와 정규화(MinMaxScaler)의 차이 — 표준화는 평균 0·표준편차 1, 정규화는 0~1 스케일입니다. 이상치에 민감할 때는 표준화가 더 안정적입니다.

---

## 8. 평가 지표

### 회귀 평가 지표

| 지표 | 의미 | 특징 |
|---|---|---|
| **RMSE** | 오차 제곱평균의 제곱근 | 단위가 원래 데이터와 같음 |
| **MAE** | 절대 오차의 평균 | 이상치에 덜 민감 |
| **R²** | 결정계수 (0~1) | 1에 가까울수록 설명력 높음 |

### 분류 평가 지표

| 지표 | 의미 |
|---|---|
| **정확도 (Accuracy)** | 전체 중 맞게 예측한 비율 — 불균형 데이터에 취약 |
| **정밀도 (Precision)** | True로 예측한 것 중 실제 True 비율 |
| **재현율 (Recall)** | 실제 True 중 모델이 True로 맞춘 비율 |
| **F1-score** | 정밀도와 재현율의 조화평균 |
| **ROC-AUC** | 임계값 변화에 따른 성능 곡선의 면적 |

정밀도와 재현율은 트레이드오프 관계입니다. 암 진단처럼 False Negative(실제 암인데 정상이라고 예측)가 치명적인 경우에는 재현율을 높이는 방향으로, 스팸 필터처럼 False Positive(정상인데 스팸이라고 예측)가 불편한 경우에는 정밀도를 높이는 방향으로 임계값을 조정합니다.

---

## 9. 마무리

### 핵심 요약

| 단계 | 도구 | 핵심 포인트 |
|---|---|---|
| EDA | Pandas, Seaborn | 탐색만, 수정 X |
| 데이터 분할 | `train_test_split` | test는 마지막 한 번만 |
| 표준화 | `StandardScaler` | test에 fit 절대 금지 |
| 학습/예측 | `model.fit / predict` | 모든 Scikit-learn 모델 공통 구조 |

| 구분 | 표준화 | 정규화 |
|---|---|---|
| 결과 | 평균 0, 표준편차 1 | 0~1 스케일 |
| Scikit-learn | `StandardScaler` | `MinMaxScaler` |

### 느낀 점

EDA를 배우기 전에는 "데이터 받으면 바로 모델 돌리면 되는 거 아닌가"라고 생각했는데, 히트맵이나 pairplot을 찍어보면 어떤 변수가 중요한지, 어떤 변수가 서로 비슷한 정보를 담고 있는지가 한눈에 보입니다. 표준화 때 데이터 누수 개념도 처음에는 직관적으로 이해가 안 됐는데, "테스트 데이터의 통계값을 학습 과정에서 쓰면 안 된다"고 생각하니 자연스럽게 와닿았습니다.

### day4 예고

다음은 **신경망(Neural Network)과 MLP** 파트입니다. day2에서 경사하강법을 직접 구현해봤고, day3에서 Scikit-learn 파이프라인을 익혔으니, day4에서는 그 기반 위에 PyTorch로 신경망을 구성하는 내용이 나옵니다.

이 글이 도움이 됐다면 댓글이나 공감 부탁드립니다. 틀린 내용이 있으면 알려주세요!

---

## 관련 포스트 추천
- [AI 입문] AI Math — 선형회귀 · 로지스틱회귀 NumPy 직접 구현 (day2)
- [AI 입문] 신경망(Neural Network)과 MLP — PyTorch 입문 (day4)
- EDA 완전 정복 — 상관관계 분석부터 pairplot까지
- 데이터 누수(Data Leakage)란? — ML 모델 평가의 함정

---

#EDA #Scikit-learn #데이터전처리 #표준화 #머신러닝파이프라인 #공부기록
