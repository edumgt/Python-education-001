# TimeSeriesWindow.py 코드 설명

> 한 줄 요약: 과거 N일치 주가를 하나의 "창문"처럼 잘라서 LSTM 신경망에 넣고, M일 후의 주가를 예측하는 코드입니다.

---

## 이 코드가 하는 일

책을 읽을 때 한 페이지씩 읽는 것이 아니라, 책갈피를 끼우고 "최근 20페이지"씩 묶어서 읽는 것과 비슷합니다. 이 코드는 주가 데이터를 "윈도우(창문)" 크기만큼 잘라서 LSTM 신경망에 넣고, 며칠 후의 주가를 예측합니다. 예를 들어 "지난 20일치 주가를 보고 5일 후 주가를 맞혀봐!"라고 학습시키는 것입니다. 사용자가 직접 윈도우 크기, 예측 기간, 종목 코드를 입력할 수 있어서 다양한 상황에 적용할 수 있습니다.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `os` | 결과 폴더(`result`) 생성 |
| `sys` | 터미널 입력(stdin)을 감지합니다 |
| `time` | 단계별 출력 사이 잠깐 대기 |
| `argparse` | 터미널에서 `--ticker AAPL` 같은 옵션을 받을 수 있게 합니다 |
| `matplotlib.pyplot` | 그래프를 그리고 저장합니다 |
| `numpy` | 숫자 배열 계산 |
| `korean_font` | 그래프 한글 글꼴 설정 |
| `torch / torch.nn` | PyTorch: LSTM 딥러닝 모델을 만들고 학습시킵니다 |
| `yfinance` | 인터넷에서 실제 주가 데이터를 내려받습니다 |

---

## 코드 흐름 (단계별 설명)

### 1단계: 사용자 입력 받기 (argparse + 대화형 입력)
```python
parser.add_argument('--ticker',  type=str, default=None)
parser.add_argument('--window',  type=int, default=None)
parser.add_argument('--horizon', type=int, default=None)
TICKER  = args.ticker  or ask_str("종목 티커", "078935.KS")
WINDOW  = args.window  or ask_int("윈도우 크기 (일)", 20, 5, 120)
HORIZON = args.horizon or ask_int("예측 기간 (일)", 5, 1, 30)
```
> 📌 **쉬운 설명:** 터미널에서 `python3 TimeSeriesWindow.py --ticker AAPL --window 30 --horizon 5`처럼 직접 옵션을 줄 수도 있고, 그냥 실행하면 대화형으로 물어봅니다. 기본값은 GS피앤엘(078935.KS), 20일 윈도우, 5일 후 예측입니다.

---

### 2단계: 주가 데이터 가져오기 & 정규화
```python
prices = df['Close'].squeeze().dropna().values
norm = (prices - p_min) / (p_max - p_min + 1e-8)
```
> 📌 **쉬운 설명:** 주가를 가져온 다음, 모든 값을 0과 1 사이로 압축합니다(Min-Max 정규화). 예를 들어 주가가 1만 원~5만 원 사이라면, 1만 원은 0.0, 5만 원은 1.0, 3만 원은 0.5가 됩니다. 이렇게 하면 LSTM이 숫자 크기에 상관없이 더 잘 학습할 수 있습니다.

---

### 3단계: 슬라이딩 윈도우 데이터셋 만들기
```python
while i + WINDOW + HORIZON <= len(norm):
    X_list.append(norm[i:i + WINDOW])           # 과거 WINDOW일치
    y_list.append(norm[i + WINDOW + HORIZON - 1])  # HORIZON일 후 값
    i += STEP
```
> 📌 **쉬운 설명:** 책의 첫 20페이지를 읽고 25페이지를 예측하고, 다음에는 2~21페이지를 읽고 26페이지를 예측하고... 이렇게 창문을 한 칸씩 옆으로 밀면서 계속 문제를 만듭니다. 이것이 "슬라이딩 윈도우"입니다.

---

### 4단계: LSTM 모델 만들기
```python
class WindowLSTM(nn.Module):
    def __init__(self, win, hidden=48):
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```
> 📌 **쉬운 설명:** LSTM은 RNN의 업그레이드 버전입니다. 기억을 "오래 기억할 것(장기 기억)"과 "잠깐 기억할 것(단기 기억)"으로 나누어 관리하기 때문에 오래된 과거도 잊지 않습니다. 2층으로 쌓인 LSTM이 20일치 주가를 읽고 마지막에 Linear(fc) 층이 최종 예측값 하나를 출력합니다. `dropout=0.2`는 20% 확률로 뉴런을 꺼서 과적합을 막습니다.

---

### 5단계: 모델 학습 (160 에폭, 배치 32개)
```python
for epoch in range(EPOCHS):
    for s in range(0, len(X_tr), BATCH):
        pred = model(xb)
        loss = crit(pred, yb)    # MSE 손실
        loss.backward()          # 역전파
        opt.step()               # 가중치 업데이트
    sched.step()                 # 학습률 조정
```
> 📌 **쉬운 설명:** 160번 반복하면서 32개씩 묶어서 학습합니다. MSE(평균 제곱 오차)로 오류를 계산하고, `loss.backward()`로 오류를 뒤로 전파해서 가중치를 조금씩 고칩니다. `sched.step()`은 80번마다 학습률을 절반으로 줄여서 처음에는 빠르게, 나중에는 섬세하게 학습합니다.

---

### 6단계: 테스트 평가 & 미래 예측
```python
mae  = np.mean(np.abs(pred_real - true_real))
rmse = np.sqrt(np.mean((pred_real - true_real)**2))
future_price = future_norm * (p_max - p_min) + p_min
print(f"{HORIZON}일 후 예측: {future_price:.2f} ({'▲ 상승' if future_price > prices[-1] else '▼ 하락'})")
```
> 📌 **쉬운 설명:** 테스트 데이터에서 예측값과 실제값의 차이(MAE, RMSE)를 계산합니다. 그리고 가장 최근 20일치 주가를 모델에 넣어서 실제로 며칠 후 가격이 얼마일지 예측합니다. 0~1로 압축했던 값을 원래 가격 범위로 다시 되돌립니다(역정규화).

---

### 7단계: 4개 패널 그림 저장
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 왼쪽 위: 윈도우 샘플 5개
# 오른쪽 위: 학습 손실
# 왼쪽 아래: 예측 vs 실제
# 오른쪽 아래: 실행 설정 요약
plt.savefig(f"../result/TimeSeriesWindow_{TICKER}_w{WINDOW}_h{HORIZON}.png")
```
> 📌 **쉬운 설명:** 2×2 격자에 4개의 그래프를 그립니다. 어떤 윈도우 모양으로 학습했는지, 학습이 잘 됐는지, 예측이 얼마나 정확한지, 그리고 설정 요약을 한눈에 볼 수 있게 저장합니다.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| **슬라이딩 윈도우** | 긴 주가 데이터에서 창문 크기만큼 잘라서 학습 데이터를 만드는 방법. 창문을 한 칸씩 오른쪽으로 밀면서 반복합니다 |
| **윈도우 크기 (window)** | 모델이 한 번에 보는 과거 날짜 수. 20이면 "지난 20일치를 보고 예측" |
| **예측 기간 (horizon)** | 몇 일 후를 예측할지. 5이면 "5일 후 가격 예측" |
| **스텝 (step)** | 창문을 한 번에 몇 칸씩 이동할지. 1이면 매일 이동, 2이면 이틀씩 건너뜁니다 |
| **LSTM** | Long Short-Term Memory. RNN의 업그레이드 버전으로 장기 기억과 단기 기억을 따로 관리해서 오래된 정보도 기억합니다 |
| **Min-Max 정규화** | 모든 값을 0~1 사이로 변환하는 것. (현재값 - 최솟값) ÷ (최댓값 - 최솟값) |
| **역정규화** | 0~1로 압축한 값을 다시 원래 가격으로 되돌리는 것 |
| **MSE (평균 제곱 오차)** | (예측값 - 실제값)²의 평균. 오차가 클수록 큰 패널티를 줍니다 |
| **MAE (평균 절대 오차)** | |예측값 - 실제값|의 평균. 실제 가격 차이가 평균 얼마나 나는지 알려줍니다 |
| **RMSE** | MSE의 제곱근. MAE와 비슷하지만 큰 오차를 더 크게 벌합니다 |
| **배치 (batch)** | 한 번에 모아서 학습하는 데이터 묶음의 크기. 32이면 32개를 한꺼번에 계산합니다 |
| **드롭아웃 (dropout)** | 학습 중에 뉴런 일부를 무작위로 끄는 기법. 특정 패턴에만 지나치게 의존하는 것(과적합)을 막습니다 |
| **학습률 스케줄러** | 학습이 진행될수록 학습률을 자동으로 줄이는 장치. 처음엔 크게, 나중엔 섬세하게 조정합니다 |
| **argparse** | 터미널에서 `--ticker AAPL`처럼 옵션을 전달받는 파이썬 도구 |
| **PyTorch** | 딥러닝 모델을 만들고 학습시키는 파이썬 라이브러리 |

---

## 실행 결과

- **파일:** `result/TimeSeriesWindow_078935_KS_w20_h5.png` (설정에 따라 파일명 변경)
- **왼쪽 위 그래프:** 슬라이딩 윈도우 샘플 5개의 모양 (정규화된 값)
- **오른쪽 위 그래프:** 160번 학습 동안 MSE 손실이 줄어드는 곡선
- **왼쪽 아래 그래프:** 테스트 데이터에서 실제 가격(파란 실선)과 LSTM 예측(빨간 점선) 비교
- **오른쪽 아래:** 실행 설정 요약표 (종목, 윈도우, 예측 기간, MAE, RMSE, 현재 가격, 예측 가격)
- **터미널 출력:** 현재 가격과 N일 후 예측 가격 및 상승/하락 방향

### CLI 사용 예시
```bash
# 애플 주식, 30일 윈도우, 5일 후 예측
python3 TimeSeriesWindow.py --ticker AAPL --window 30 --horizon 5 --step 1

# 삼성전자, 기본 설정으로 실행
python3 TimeSeriesWindow.py --ticker 005930.KS
```
