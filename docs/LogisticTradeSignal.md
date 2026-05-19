# LogisticTradeSignal.py 코드 설명

> 한 줄 요약: RSI와 MACD라는 두 가지 주식 지표를 보고 "지금 사야 할까(매수)? 팔아야 할까(매도)?"를 확률로 알려주는 분류 모델을 만드는 코드입니다.

---

## 이 코드가 하는 일

주식을 살 때 "지금 사면 오를까, 내릴까?" 가 제일 궁금하죠? 이 코드는 RSI(상대강도지수)와 MACD(이동평균수렴확산지수)라는 두 가지 신호등을 보고, 내일 주가가 오를 확률이 높은지 낮은지를 0~1 사이의 숫자로 알려줘요.

예를 들어 확률이 0.8이면 "내일 오를 확률이 80%", 0.2이면 "내일 내릴 확률이 80%"예요. 이것은 Yes/No로 분류하는 문제라서 **로지스틱 회귀**라는 방법을 써요. 실제 주식 데이터(GS피앤엘)를 받아서 사용하고, 인터넷이 안 되면 가상 데이터로 실습해요.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `os` | 결과 폴더를 만들어요 |
| `time` | 단계마다 잠깐 기다려줘요 |
| `matplotlib.pyplot` | 그래프를 그려줘요 |
| `numpy` | 숫자 배열 계산을 빠르게 해줘요 |
| `pandas` | 주가 데이터를 표 형식으로 다루고, RSI/MACD 계산에도 써요 |
| `korean_font` | 그래프에서 한국어가 안 깨지도록 설정해요 |
| `sklearn.linear_model.LogisticRegression` | 확률로 분류하는 로지스틱 회귀 모델이에요 |
| `sklearn.metrics.classification_report` | 정확도, 정밀도, 재현율 등을 표 형식으로 보여줘요 |
| `sklearn.metrics.roc_auc_score` | 모델의 분류 성능을 0~1 사이 점수로 나타내줘요 |
| `sklearn.model_selection.train_test_split` | 데이터를 학습용과 테스트용으로 나눠줘요 |
| `sklearn.preprocessing.StandardScaler` | 숫자 크기를 통일해줘요 |
| `yfinance` | 인터넷에서 실제 주가 데이터를 내려받아줘요 |

---

## 코드 흐름 (단계별 설명)

### 1단계: RSI와 MACD 계산 함수 만들기

```python
def compute_rsi(prices, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + rs)).values

def compute_macd(prices, fast=12, slow=26):
    return (s.ewm(span=fast).mean() - s.ewm(span=slow).mean()).values
```

> 📌 **쉬운 설명:** RSI는 "최근 14일 동안 오른 힘과 내린 힘을 비교해서 0~100 사이 숫자로 나타낸 것"이에요. 70 이상이면 주가가 과하게 오른 상태(과열), 30 이하면 너무 내린 상태(과매도)예요. MACD는 빠른 평균과 느린 평균의 차이로, 주가가 오르는 추세인지 내리는 추세인지 알려줘요.

---

### 2단계: 실제 데이터 준비 & 정답 라벨 만들기

```python
rsi_feat = rsi_full[:-1]     # 오늘 RSI
macd_feat = macd_full[:-1]   # 오늘 MACD
y_all = (returns_full > 0).astype(int)  # 내일 오르면 1(매수), 아니면 0
```

> 📌 **쉬운 설명:** 오늘의 RSI와 MACD를 보고 "내일 주가가 오를까?"를 예측하는 게 목표예요. 내일 실제로 오르면 정답 라벨이 1(매수), 내리거나 그대로면 0(매도/관망)이에요.

---

### 3단계: 데이터 나누기와 표준화

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

> 📌 **쉬운 설명:** 전체 데이터의 75%로 공부하고 25%로 시험을 봐요. 그리고 RSI(20~80)와 MACD(-3~3)는 숫자 범위가 달라서 표준화로 같은 기준으로 맞춰줘요.

---

### 4단계: 로지스틱 회귀 모델 학습

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
# 수식: P(매수) = sigmoid(w0 + w1×RSI + w2×MACD)
```

> 📌 **쉬운 설명:** RSI와 MACD를 조합해서 "매수 확률"을 계산하는 공식을 배워요. **sigmoid 함수**가 어떤 숫자든 0~1 확률로 바꿔줘요. 확률이 0.5 이상이면 "매수(1)", 이하면 "매도/관망(0)"으로 결정해요.

---

### 5단계: 모델 평가

```python
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(classification_report(y_test, y_pred, target_names=['매도/관망', '매수']))
```

> 📌 **쉬운 설명:** 시험 데이터로 "정확도", "정밀도", "재현율" 등을 계산해요. ROC-AUC 점수는 모델이 얼마나 잘 분류하는지를 나타내는 점수예요. 0.5는 동전 던지기 수준, 1.0은 완벽한 예측이에요.

---

### 6단계: 결정 경계 그래프 그리기

```python
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)
cf = ax.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.4)
```

> 📌 **쉬운 설명:** RSI와 MACD의 모든 조합에서 매수 확률을 계산해서 색깔 지도를 그려요. 초록색 구역(오른쪽 위)은 매수 확률이 높고, 빨간색 구역(왼쪽 아래)은 낮아요. 그 위에 실제 매수/매도 점들을 찍어서 모델이 얼마나 잘 나누는지 볼 수 있어요.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| 로지스틱 회귀 (Logistic Regression) | 숫자를 예측하는 선형 회귀와 달리, "네/아니오"처럼 두 종류로 분류하는 방법이에요. 확률(0~1)을 계산해줘요. |
| RSI (상대강도지수) | 최근 14일의 "오른 힘 vs 내린 힘" 비교 지표예요. 70 이상은 과열, 30 이하는 침체 신호예요. |
| MACD | 빠른 이동평균에서 느린 이동평균을 뺀 값이에요. 양수면 오르는 추세, 음수면 내리는 추세예요. |
| sigmoid 함수 | 어떤 숫자든 0~1 사이로 바꿔주는 수학 함수예요. 0.5 기준으로 분류를 결정해요. |
| 결정 경계 | "매수"와 "매도/관망"을 나누는 경계선이에요. 이 선 오른쪽은 매수, 왼쪽은 매도예요. |
| 정밀도 (Precision) | "매수라고 예측한 것 중에 실제로 매수가 맞는 비율"이에요. 틀린 매수 신호가 적을수록 높아요. |
| 재현율 (Recall) | "실제 매수 신호 중에서 모델이 잡아낸 비율"이에요. 놓친 매수 기회가 적을수록 높아요. |
| ROC-AUC | 분류 모델의 종합 성능 점수예요. 0.5는 무작위 수준, 1.0은 완벽한 분류예요. |
| 매수 (Buy) | 주식을 사는 것이에요. 앞으로 오를 것 같을 때 해요. |
| 매도/관망 | 주식을 팔거나 아무것도 안 하는 것이에요. 앞으로 내릴 것 같을 때 해요. |
| 표준화 | 서로 다른 범위의 숫자들을 평균 0, 표준편차 1로 맞추는 것이에요. |
| `ewm` (지수가중이동평균) | 최근 데이터에 더 높은 비중을 두는 평균 계산 방법이에요. 오래된 데이터는 영향이 작아요. |

---

## 실행 결과

- **파일:** `result/LogisticTradeSignal_078935_KS.png`
- **내용:** RSI(가로축)와 MACD(세로축)를 기준으로 한 매수/매도 분류 지도
  - 초록색 배경: 매수 확률이 높은 구역 (오른쪽 위 — RSI 높고 MACD 양수)
  - 빨간색 배경: 매도/관망 확률이 높은 구역 (왼쪽 아래 — RSI 낮고 MACD 음수)
  - 빨간 점: 실제로 다음 날 주가가 오른 날(매수 신호)
  - 파란 점: 실제로 다음 날 주가가 내리거나 그대로인 날(매도/관망 신호)
  - 오른쪽 컬러바: 색깔이 매수 확률을 나타내요 (0 = 매도, 1 = 매수)
