# TransformerAttention.py 코드 설명

> 한 줄 요약: 주식 가격 데이터를 보고 "내일 주가가 오를까, 내릴까?"를 트랜스포머(Transformer) AI가 예측하는 프로그램입니다.

---

## 이 코드가 하는 일

책을 읽을 때 중요한 단어에 밑줄을 긋는 것처럼, 이 AI는 과거 30일치 주가 변화 중에서 "어느 날이 가장 중요한지"를 스스로 찾아냅니다.
그 능력을 **Attention(어텐션)** 이라고 부르는데, 마치 시험 전날 밤에 노트에서 꼭 봐야 할 부분에 형광펜을 칠하는 것과 같습니다.
AI는 과거 30일의 수익률 패턴을 보고 다음 날 주가가 오를지(1) 내릴지(0)를 맞히도록 학습합니다.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `os` | 컴퓨터 폴더를 만들거나 파일 경로를 다루는 도구 |
| `time` | 잠깐 기다리게 하는 도구 (화면 출력 사이에 짧은 휴식) |
| `math` | 수학 계산 도구 (로그, 제곱근 등) |
| `matplotlib.pyplot` | 그래프를 그리는 도구 (그림판 같은 것) |
| `numpy` | 숫자 배열을 빠르게 계산하는 도구 (수학 계산기) |
| `torch` | 딥러닝(AI 두뇌 학습) 전용 도구 |
| `torch.nn` | 신경망(뉴런 연결망)을 만드는 부품 모음 |
| `korean_font` | 그래프에 한글이 깨지지 않도록 폰트를 설정하는 도구 |
| `yfinance` | 인터넷에서 실제 주식 가격을 가져오는 도구 |

---

## 코드 흐름 (단계별 설명)

### 1단계: 주가 데이터 가져오기
```python
df = yf.download('078935.KS', start='2020-01-01', ...)
prices = df['Close'].squeeze().dropna().values
```
> 📌 **쉬운 설명:** 인터넷에서 "GS피앤엘"이라는 회사의 주식 가격을 2020년부터 오늘까지 가져옵니다. 만약 인터넷 연결이 안 되면 컴퓨터가 가짜 주가 데이터를 직접 만들어서 사용합니다.

---

### 2단계: 슬라이딩 윈도우로 문제 만들기
```python
SEQ_LEN = 30
for i in range(len(returns) - SEQ_LEN):
    X_list.append(returns[i:i + SEQ_LEN])    # 과거 30일
    y_list.append(1 if returns[i + SEQ_LEN] > 0 else 0)  # 내일 오름/내림
```
> 📌 **쉬운 설명:** "30칸짜리 창문"을 하루씩 앞으로 밀면서 문제를 만드는 것입니다. 창문 안의 30일 수익률이 문제(입력), 창문 바로 다음 날의 방향이 정답(출력)이 됩니다. 마치 30일치 일기를 보고 "내일 날씨를 맞혀라" 하는 퀴즈를 수백 개 만드는 것과 같습니다.

---

### 3단계: 위치 정보 넣기 (Positional Encoding)
```python
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
```
> 📌 **쉬운 설명:** 트랜스포머는 기본적으로 순서를 모릅니다. "1일째", "2일째"라는 개념이 없는 것입니다. 그래서 sin(사인)과 cos(코사인) 파도 모양 숫자를 각 날짜에 더해줘서 "이게 몇 번째 날이에요"라는 위치 힌트를 줍니다. 마치 줄 세운 아이들에게 번호표를 달아주는 것과 같습니다.

---

### 4단계: 트랜스포머 AI 모델 만들기
```python
class StockTransformer(nn.Module):
    self.input_proj = nn.Linear(input_dim, d_model)   # 입력 변환
    self.pos_enc    = PositionalEncoding(...)          # 위치 정보
    self.encoder    = nn.TransformerEncoder(...)       # 핵심 두뇌
    self.classifier = nn.Sequential(...)               # 최종 판단
```
> 📌 **쉬운 설명:** AI의 두뇌를 설계하는 단계입니다. 공장 조립 라인처럼 데이터가 순서대로 처리됩니다. ① 숫자를 크게 펼치고(Linear) → ② 위치 힌트를 더하고(PositionalEncoding) → ③ 어느 날짜가 중요한지 파악하고(TransformerEncoder) → ④ 오를지 내릴지 최종 판단합니다(classifier).

---

### 5단계: 학습 설정
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
EPOCHS, BATCH = 120, 32
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```
> 📌 **쉬운 설명:** AI가 공부하는 방법을 정합니다. `CrossEntropyLoss`는 AI가 얼마나 틀렸는지 점수를 매기는 채점 기준표입니다. `Adam`은 틀린 부분을 어떻게 고칠지 결정하는 공부 방법입니다. 120번 전체 데이터를 공부하고(에폭), 한 번에 32개씩 묶어서 공부합니다(배치).

---

### 6단계: AI 학습시키기
```python
for epoch in range(EPOCHS):
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```
> 📌 **쉬운 설명:** AI가 실제로 공부하는 단계입니다. 문제를 풀고 → 틀린 정도를 계산하고 → 두뇌를 조금씩 수정합니다. `clip_grad_norm_`은 한 번에 너무 크게 바꾸지 않도록 하는 안전장치입니다. 마치 연필로 쓰다가 너무 강하게 누르면 종이가 찢어지니까 힘을 조절하는 것과 같습니다.

---

### 7단계: Attention 가중치 꺼내기
```python
_, attn_w = first_layer.self_attn(x_enc, x_enc, x_enc, need_weights=True)
attn_map = attn_w[0].numpy()  # (30, 30) 크기의 표
```
> 📌 **쉬운 설명:** 학습이 끝난 AI가 테스트 데이터를 볼 때 "30일 중 어느 날을 얼마나 중요하게 봤는지"를 숫자 표로 뽑아냅니다. 이 표가 바로 Attention 히트맵이 됩니다.

---

### 8단계: 그래프로 저장하기
```python
plt.savefig("../result/TransformerAttention_078935_KS.png", dpi=150)
```
> 📌 **쉬운 설명:** 학습 과정(손실, 정확도), 예측 결과, Attention 히트맵을 하나의 그림으로 합쳐서 파일로 저장합니다.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| **Attention(어텐션)** | 책을 읽을 때 중요한 부분에 형광펜 칠하는 것처럼, AI가 "어느 날짜가 중요한지"를 스스로 결정하는 능력 |
| **Q(Query)** | "나는 지금 무엇을 예측하려 하지?" 라는 질문 |
| **K(Key)** | "나는 이런 특징을 가진 날이야" 라고 소개하는 이름표 |
| **V(Value)** | 질문과 이름표가 잘 맞을 때 실제로 전달하는 정보 내용 |
| **Multi-Head Attention** | 같은 데이터를 여러 관점에서 동시에 살펴보는 것 (여러 명이 함께 형광펜 칠하는 것) |
| **Positional Encoding** | 순서를 모르는 AI에게 "이게 몇 번째야"라고 알려주는 번호표 |
| **TransformerEncoder** | Attention을 여러 겹 쌓아 만든 AI의 핵심 두뇌 부분 |
| **슬라이딩 윈도우** | 창문을 하루씩 밀면서 문제와 정답을 만드는 방법 |
| **수익률(return)** | 어제보다 오늘 주가가 몇 % 변했는지를 나타내는 숫자 |
| **에폭(Epoch)** | AI가 전체 문제를 처음부터 끝까지 한 번 다 푸는 것 |
| **배치(Batch)** | 한꺼번에 공부하는 문제 묶음의 크기 |
| **손실(Loss)** | AI가 틀린 정도를 나타내는 점수. 낮을수록 좋음 |
| **CrossEntropyLoss** | "오를 것이다"와 "내릴 것이다" 중 하나를 고르는 문제의 채점 방식 |
| **Adam 옵티마이저** | AI 두뇌를 어떻게 조금씩 고쳐나갈지 결정하는 공부 전략 |
| **Softmax** | 여러 점수를 확률(모두 더하면 100%)로 바꾸는 함수 |
| **히트맵(Heatmap)** | 숫자 표를 색깔로 표현한 그림 (밝을수록 숫자가 큼) |
| **Global Average Pooling** | 긴 시퀀스 결과를 하나의 숫자 묶음으로 압축하는 것 (반 전체 점수의 평균을 내는 것) |
| **파라미터(Parameter)** | AI 두뇌 속에 있는 수많은 숫자들 — 학습으로 조정됨 |

---

## 실행 결과

- **저장 파일:** `result/TransformerAttention_078935_KS.png`
- 그림 안에 총 4개의 그래프가 만들어집니다:
  1. **학습 손실 그래프** — AI가 공부할수록 오답률이 줄어드는 모습
  2. **학습 정확도 그래프** — AI가 공부할수록 맞히는 비율이 늘어나는 모습 (테스트 정확도도 표시)
  3. **상승 예측 확률 막대그래프** — 테스트 데이터에서 AI가 각 날을 "오를 것이다"라고 얼마나 확신했는지
  4. **Self-Attention 히트맵** — 30×30 표로, 밝은 칸일수록 두 날짜가 서로 "관련 있다"고 AI가 판단한 것
