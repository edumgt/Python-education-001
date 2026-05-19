# CnnLstmHybrid.py 코드 설명

> 한 줄 요약: 짧은 구간 패턴을 잘 찾는 CNN과, 긴 흐름을 기억하는 LSTM을 합쳐서 주가의 오름/내림을 예측하는 코드예요.

---

## 이 코드가 하는 일

사람이 주식을 공부할 때 "오늘 하루 뉴스"도 보고 "지난 2주 흐름"도 함께 보듯이, 이 AI도 두 가지 방식을 함께 사용해요.
60일치 주가 변화를 6개의 10일짜리 덩어리로 나눠서, CNN이 각 10일 덩어리의 짧은 패턴을 요약하고, LSTM이 6개 요약본의 시간 순서를 기억해요.
두 가지 능력을 합쳐서 "내일 주가가 오를까 내릴까"를 더 정확하게 예측하려는 시도예요.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `os` | 결과 저장 폴더를 만드는 도구 |
| `time` | 단계 사이에 잠깐 쉬면서 진행 상황을 보여주는 시계 |
| `matplotlib.pyplot` | 그래프를 그리는 도화지 |
| `numpy` | 숫자 배열과 빠른 수학 계산 도구 |
| `torch` | 딥러닝 AI를 만들고 훈련시키는 핵심 도구 (파이토치) |
| `torch.nn` | CNN·LSTM·선형 레이어 등 AI 부품들의 모음 |
| `korean_font` | 그래프에 한글을 쓸 수 있게 해주는 글꼴 도구 |
| `yfinance` | 인터넷에서 실제 주가 데이터를 받아오는 도구 |

---

## 코드 흐름 (단계별 설명)

### 1단계: 주가 데이터 불러오기
```python
df = yf.download('078935.KS', start='2020-01-01', ...)
prices = df['Close'].values.astype(np.float32)
returns = np.diff(prices) / prices[:-1]
```
> **쉬운 설명:** GS피앤엘 주식의 종가를 가져와요. 그런 다음 "오늘 주가가 어제보다 몇 퍼센트 올랐나?" 하는 일간 수익률로 변환해요. 예를 들어 어제 100원이었는데 오늘 102원이라면 수익률은 +0.02(+2%)예요.

---

### 2단계: 60일을 6개 × 10일 덩어리로 나누기
```python
N_WINDOWS = 6    # LSTM이 볼 덩어리 수
WIN_SIZE  = 10   # 각 덩어리의 길이 (10일)
seq = returns[i:i + 60]                  # 60일치 수익률
windows = seq.reshape(N_WINDOWS, WIN_SIZE)  # → (6, 10) 모양으로 나눔
```
> **쉬운 설명:** 60일치 수익률을 10일씩 6덩어리로 잘라요. 마치 두꺼운 책(60페이지)을 10페이지짜리 챕터 6개로 나누는 것과 같아요. CNN은 각 챕터를 읽고 요약하고, LSTM은 6개 챕터 요약본을 순서대로 읽어요.

---

### 3단계: CNN + LSTM 하이브리드 모델 만들기
```python
class CnnLstmHybrid(nn.Module):
    def __init__(self, ...):
        # CNN 인코더: 각 10일 덩어리에서 패턴 추출
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 10일 → 숫자 32개로 압축
        )
        # LSTM: 6개의 요약 벡터를 순서대로 기억
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2)
        self.classifier = nn.Linear(64, 2)  # 상승/하락 선택
```
> **쉬운 설명:** 이 모델은 두 개의 전문가가 팀을 이루고 있어요. 첫 번째 전문가 CNN은 "이번 주(10일) 패턴을 보니 상승세네!"처럼 짧은 구간을 분석해서 숫자 32개짜리 요약 카드를 만들어요. 두 번째 전문가 LSTM은 6장의 요약 카드를 시간 순서대로 읽으며 "지난 6주 흐름을 종합하면 다음 주는 오를 것 같아!"라고 판단해요.

---

### 4단계: 모델 내부 흐름 (forward)
```python
def forward(self, x):
    B, N, W = x.shape              # 배치 크기, 6, 10
    x_cnn = x.view(B * N, 1, W)   # 모든 10일 덩어리를 펼침
    feat  = self.cnn(x_cnn)        # CNN으로 각 덩어리 압축
    feat  = feat.view(B, N, -1)    # 다시 배치별로 묶음 → (배치, 6, 32)
    out, _ = self.lstm(feat)       # LSTM이 순서대로 읽음
    return self.classifier(out[:, -1, :])  # 마지막 기억으로 분류
```
> **쉬운 설명:** 데이터가 모델 안으로 들어오면 이렇게 처리돼요: ① 6×10 표를 10짜리 행 6개로 분리 → ② CNN이 각 행을 32개 숫자로 압축 → ③ 6개의 32숫자짜리 카드를 LSTM에게 순서대로 전달 → ④ LSTM이 마지막에 가장 최신 기억을 꺼내서 "상승 or 하락"을 결정해요.

---

### 5단계: 학습 설정 (고급 설정 사용)
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
```
> **쉬운 설명:** 세 가지 특별한 장치를 써요. `label_smoothing=0.1`은 AI가 정답을 너무 확신하지 않도록 살짝 겸손하게 만드는 장치예요. `Adam`은 AI가 배우는 속도를 영리하게 조절해요. `CosineAnnealingLR`은 학습 속도를 처음엔 크게, 나중엔 조금씩 줄여가는 스케줄러예요 — 마치 달리기할 때 처음엔 빠르게 뛰다가 결승선 가까이서 정밀하게 걷는 것처럼요.

---

### 6단계: 학습 루프
```python
for epoch in range(60):
    out  = model(xb)
    loss = criterion(out, yb)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```
> **쉬운 설명:** 60번 반복하며 배워요. `clip_grad_norm_`은 AI가 한 번에 너무 크게 바뀌지 않도록 조절하는 안전장치예요 (급변 방지). 실수를 계산하고(backward), 파라미터를 수정하고(step), 학습률을 조절하는(scheduler) 세 동작을 반복해요.

---

### 7단계: 결과 시각화 및 저장
```python
ax3.bar(range(n_show), probs[:n_show], color=colors, ...)
plt.savefig("../result/CnnLstmHybrid_078935_KS.png")
```
> **쉬운 설명:** 테스트 데이터에 대한 "상승 예측 확률"을 막대그래프로 보여줘요. 확률이 전체 중앙값보다 높으면 빨간 막대(상승 예측), 낮으면 파란 막대(하락 예측)로 표시해요.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| CNN (1D) | 연속된 숫자 배열에서 짧은 구간 패턴을 찾는 AI 레이어 |
| LSTM | "Long Short-Term Memory". 오래된 것도 잊지 않고 기억하는 특별한 AI 레이어 |
| 하이브리드 | 두 가지 다른 모델을 합쳐서 각각의 장점을 함께 쓰는 방법 |
| Conv1d | 1차원(시계열) 데이터에서 패턴을 찾는 필터. 3일치 패턴을 훑으며 분석해요 |
| AdaptiveAvgPool1d(1) | 시계열 데이터를 딱 1개의 숫자 묶음으로 압축하는 레이어 |
| 수익률 | 주가 변화를 퍼센트로 나타낸 것. (오늘 - 어제) ÷ 어제 × 100% |
| z-score 정규화 | 숫자들을 평균 0, 흔들림 1 기준으로 맞춰서 비교하기 쉽게 만드는 것 |
| 슬라이딩 윈도우 | 창문을 하루씩 밀면서 60일짜리 구간을 잘라내는 방식 |
| Label Smoothing | "정답이 100% 확실해"라는 과자신감을 줄여주는 학습 기법 |
| CosineAnnealingLR | 학습 속도를 코사인 곡선처럼 부드럽게 줄여가는 스케줄러 |
| clip_grad_norm | 기울기가 너무 커지는 것을 막는 안전장치. 폭주 방지 |
| Dropout | 학습 중 일부 뉴런을 꺼서 모델이 외우기보다 이해하도록 하는 기법 |
| 은닉층 (hidden) | LSTM 내부에서 정보를 처리하고 전달하는 중간 레이어 |
| softmax | 출력 숫자들을 "확률"로 변환하는 함수. 합이 항상 1이에요 |

---

## 실행 결과

- **파일:** `result/CnnLstmHybrid_078935_KS.png`
- **내용:** 2행 2열 + 하단 패널로 구성된 그래프 이미지
  - **왼쪽 위:** 학습이 진행될수록 손실(오차)이 줄어드는 그래프. 처음엔 틀리다가 나중엔 잘 맞혀가는 모습을 볼 수 있어요
  - **오른쪽 위:** 학습 정확도 변화 그래프 + 최종 테스트 정확도
  - **하단:** 테스트 샘플 80개에 대한 상승 예측 확률 막대그래프 (빨강=상승 예측, 파랑=하락 예측)
  - **왼쪽 (작은 상자):** 모델 구조(CNN+LSTM 레이어 구성)를 정리한 텍스트 박스
- **콘솔 출력:** 데이터 로드 현황, 하이브리드 구조 설명, 각 에폭별 손실과 정확도, 최종 테스트 정확도
