# CnnTimeSeriesFeature.py 코드 설명

> 한 줄 요약: 1D CNN을 사용해서 주가 수익률의 연속 패턴(예: "3일 연속 상승")을 자동으로 찾아내고, 내일 주가가 오를지 내릴지 예측하는 코드예요.

---

## 이 코드가 하는 일

마치 탐정이 단서를 모아 사건을 해결하듯, 이 코드는 지난 30일간의 주가 변화 패턴을 분석해서 다음 날 주가 방향을 예측해요.
AI가 스스로 "이런 패턴이 나타나면 다음 날 오르더라!"와 같은 규칙을 학습 데이터에서 자동으로 발견해요.
학습이 끝나면 AI의 "눈"이 어느 날짜에 강하게 반응하는지도 그래프로 시각화해서 확인할 수 있어요.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `os` | 결과 저장 폴더를 만드는 도구 |
| `time` | 단계 사이에 잠깐 쉬면서 진행 상황을 보여주는 시계 |
| `matplotlib.pyplot` | 그래프를 그리는 도화지 |
| `numpy` | 숫자 배열과 빠른 수학 계산 도구 |
| `torch` | 딥러닝 AI를 만들고 훈련시키는 핵심 도구 (파이토치) |
| `torch.nn` | CNN 레이어, 활성화 함수 등 AI 부품들의 모음 |
| `korean_font` | 그래프에 한글을 쓸 수 있게 해주는 글꼴 도구 |
| `yfinance` | 인터넷에서 실제 주가 데이터를 받아오는 도구 |

---

## 코드 흐름 (단계별 설명)

### 1단계: 주가 데이터 불러오기 & 수익률 계산
```python
df = yf.download('078935.KS', start='2020-01-01', ...)
prices = df['Close'].values.astype(np.float32)
returns = np.diff(prices) / prices[:-1]
```
> **쉬운 설명:** GS피앤엘 주식의 종가를 가져와요. 그다음 "오늘 주가는 어제보다 몇 퍼센트 달라졌나?"를 계산해서 일간 수익률로 변환해요. 예를 들어 100원 → 103원이면 +0.03(+3%)이에요. 주가 절댓값 대신 수익률을 쓰면 가격 크기에 상관없이 패턴을 비교하기 쉬워요.

---

### 2단계: 슬라이딩 윈도우로 학습 데이터 만들기
```python
SEQ_LEN = 30  # 30일치를 한 묶음으로 사용
for i in range(len(returns) - SEQ_LEN):
    seq = returns[i:i + SEQ_LEN]     # 30일치 수익률
    seq = (seq - seq.mean()) / seq.std()  # z-score 정규화
    X_list.append(seq)
    y_list.append(1 if returns[i + SEQ_LEN] > 0 else 0)  # 다음 날 오름/내림
```
> **쉬운 설명:** 전체 데이터에서 "30일짜리 창문"을 하루씩 밀면서 잘라내요. 각 창문의 30일치 수익률이 입력(X), 그다음 날이 오르면 1, 내리면 0이 정답(y)이에요. 예를 들어 500일치 데이터가 있으면 약 470개의 (X, y) 쌍이 만들어져요. z-score 정규화는 모든 창문의 평균을 0, 크기를 1로 맞춰줘서 공평하게 비교할 수 있게 해요.

---

### 3단계: 1D CNN 모델 만들기
```python
class CNN1DStock(nn.Module):
    def __init__(self, seq_len=30):
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # 3일 패턴 16종 찾기
            nn.ReLU(),
            nn.MaxPool1d(2),                              # 시퀀스 30 → 15
            nn.Conv1d(16, 32, kernel_size=3, padding=1), # 복합 패턴 32종 찾기
            nn.ReLU(),
            nn.MaxPool1d(2),                              # 15 → 7
            nn.Conv1d(32, 64, kernel_size=3, padding=1), # 더 복잡한 패턴 64종
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),                      # 7 → 4로 고정 압축
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),            # 64×4 = 256개 숫자로 펼침
            nn.Linear(64 * 4, 64),
            nn.Dropout(0.4),
            nn.Linear(64, 2),        # 최종: 상승(1) or 하락(0)
        )
```
> **쉬운 설명:** 이 AI는 세 겹의 "눈"을 가지고 있어요. 첫 번째 눈(Conv1d 1→16)은 3일짜리 짧은 패턴 16가지를 찾고, 두 번째 눈(16→32)은 그 패턴들이 모인 더 큰 패턴 32가지를 찾고, 세 번째 눈(32→64)은 가장 복잡한 패턴 64가지를 찾아요. MaxPool1d는 찾은 패턴들 중 "가장 강한 신호"만 남겨서 데이터를 절반씩 압축해요. 마지막에 Flatten으로 죽 펼쳐서 상승/하락 2개 중 하나를 선택해요.

---

### 4단계: 특징 맵 추출 기능
```python
def extract_features(self, x):
    """Conv 블록까지만 실행해 중간 특징 벡터 반환"""
    with torch.no_grad():
        return self.conv_block(x)
```
> **쉬운 설명:** AI의 "눈"이 중간에 무엇을 보고 있는지 꺼내볼 수 있는 창문이에요. 최종 판단(상승/하락) 전에 AI가 어떤 패턴을 발견했는지 확인할 수 있어요.

---

### 5단계: 학습 루프
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
for epoch in range(60):
    out  = model(xb)
    loss = criterion(out, yb)
    loss.backward()
    optimizer.step()
    scheduler.step()
```
> **쉬운 설명:** 학습 데이터를 60번 반복해서 봐요. 매번 예측이 틀리면 얼마나 틀렸는지(손실)를 계산하고, AI 파라미터를 조금씩 고쳐요. 학습률 스케줄러 덕분에 처음엔 크게 배우다가 나중엔 세밀하게 조정해요.

---

### 6단계: 특징 맵 시각화 (Hook 사용)
```python
hook_output = {}
def hook_fn(module, inp, out):
    hook_output['conv1'] = out.detach()

hook = model.conv_block[0].register_forward_hook(hook_fn)
with torch.no_grad():
    model(sample_input)
hook.remove()
feat_map = hook_output['conv1'][0].numpy()  # (16채널, 30)
```
> **쉬운 설명:** 훈련된 AI의 첫 번째 Conv 레이어가 어느 날짜에서 강하게 반응하는지 들여다보는 방법이에요. "Hook"은 마치 AI 뇌 속에 몰래 카메라를 달아놓는 것처럼, AI가 처리하는 중간 결과값을 가로채서 꺼내와요. 이걸 그래프로 그리면 각 필터(눈)가 어느 날짜 구간을 중요하게 봤는지 알 수 있어요.

---

### 7단계: 결과 시각화 및 저장
```python
ax4.plot(feat_map[i], alpha=0.7, label=f"필터{i+1}")  # 특징 맵 8개
plt.savefig("../result/CnnTimeSeriesFeature_078935_KS.png")
```
> **쉬운 설명:** 네 가지 그래프를 한 장에 그려요: 학습 손실 변화, 학습 정확도 변화, 테스트 예측 확률 막대그래프, 첫 번째 Conv 레이어의 필터 8개가 각 날짜에 얼마나 강하게 반응했는지. 이 마지막 그래프를 보면 AI가 "어떤 날짜 패턴"을 중요하게 여기는지 짐작할 수 있어요.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| 1D CNN | 1차원(시간 순서) 데이터에서 패턴을 찾는 AI. 사진이 아닌 숫자 배열에 써요 |
| Conv1d | 작은 필터(돋보기)를 숫자 배열 위로 밀면서 패턴을 찾는 레이어 |
| kernel_size=3 | 한 번에 3칸(3일)씩 보는 필터의 크기 |
| MaxPool1d | 최댓값만 남기고 나머지를 압축해서 데이터를 절반으로 줄이는 레이어 |
| AdaptiveAvgPool1d | 입력 크기에 상관없이 원하는 크기로 줄여주는 평균 풀링 레이어 |
| Flatten | 2차원·3차원 데이터를 1차원 배열로 죽 펼치는 레이어 |
| Dropout | 학습 중 랜덤으로 뉴런을 꺼서 모델이 특정 패턴만 외우지 않게 하는 기법 |
| 수익률 (returns) | (오늘 주가 - 어제 주가) ÷ 어제 주가. 주가 변화를 비율로 나타낸 것 |
| 슬라이딩 윈도우 | 창문을 하루씩 밀면서 고정된 길이의 시퀀스를 잘라내는 방법 |
| z-score 정규화 | 평균을 빼고 표준편차로 나눠서 모든 구간을 같은 크기로 만드는 것 |
| 특징 맵 (feature map) | CNN 레이어가 입력 데이터에서 찾아낸 패턴 반응값의 배열 |
| Hook | 모델 내부를 실행하는 도중 중간 결과값을 가로채는 파이토치 기능 |
| 에폭 (Epoch) | 전체 학습 데이터를 한 번 다 학습한 것. 60 에폭 = 60번 반복 |
| 정확도 | AI가 상승/하락을 맞힌 비율. 0.55이면 100번 중 55번 맞혔다는 뜻 |
| label_smoothing | 정답을 100%라고 확신하지 않게 하여 AI가 더 일반적으로 배우게 하는 기법 |

---

## 실행 결과

- **파일:** `result/CnnTimeSeriesFeature_078935_KS.png`
- **내용:** 3행으로 구성된 그래프 이미지
  - **1행 왼쪽:** 학습 손실 그래프. 에폭이 늘수록 손실이 줄어드는 모습을 보여줘요
  - **1행 오른쪽:** 학습 정확도 그래프. 점선(0.5)보다 위에 있어야 동전 던지기보다 나은 거예요
  - **2행:** 테스트 샘플 60개에 대한 상승 예측 확률 막대그래프 (빨강=중앙값 이상, 파랑=중앙값 미만)
  - **3행:** 첫 번째 Conv1d 레이어의 필터 8개가 30일 시퀀스의 각 날짜에서 얼마나 강하게 반응하는지 보여주는 다중 선 그래프
- **콘솔 출력:** 데이터 로드 현황, 학습/테스트 세트 크기, 각 에폭별 손실과 정확도, 최종 테스트 정확도, 특징 맵 시각화 완료 메시지
