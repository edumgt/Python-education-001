import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401

os.makedirs("result", exist_ok=True)

print("=" * 65)
print("  바닐라 RNN 직접 구현 (BPTT): 주가 방향 예측")
print("=" * 65)
print()
print("  RNN 구조:")
print("  ┌─────────────────────────────────────────┐")
print("  │  h_t = tanh(Wx·x_t  +  Wh·h_(t-1) + b) │  ← 은닉 상태 업데이트")
print("  │  y_t = Wy·h_t + by                       │  ← 출력 계산")
print("  └─────────────────────────────────────────┘")
print("  핵심: 이전 시점의 은닉 상태(h_(t-1))를 현재에 재사용 → '기억'")

# ── 1. 데이터 생성 ─────────────────────────────────────────
print("\n[1/8] 주가 시계열 & 시퀀스 데이터 생성 중...")
time.sleep(0.5)
np.random.seed(42)
TICKER = '078935.KS'
prices = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    days = 300
    t = np.arange(days)
    prices = (100 + 0.08 * t + 3 * np.sin(t / 12) + np.random.normal(0, 1.0, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

# 수익률을 입력, 다음날 수익률 부호(오름=1/내림=0)를 정답으로
returns = np.diff(prices) / prices[:-1]
seq_len = 10           # RNN이 한 번에 볼 시퀀스 길이

X_list, y_list = [], []
for i in range(len(returns) - seq_len):
    X_list.append(returns[i:i + seq_len])
    y_list.append(1.0 if returns[i + seq_len] > 0 else 0.0)

X_all = np.array(X_list)                  # (샘플, seq_len)
y_all = np.array(y_list).reshape(-1, 1)  # (샘플, 1)

split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
print(f"   → 시퀀스 길이: {seq_len}일  |  학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
print(f"   → 정답: 다음 날 주가 방향 (오름=1, 내림=0)")
time.sleep(0.4)

# ── 2. 하이퍼파라미터 & 가중치 초기화 ─────────────────────
print("\n[2/8] RNN 가중치 초기화 중...")
print("   Wx  : 입력→은닉  (input_size × hidden_size)")
print("   Wh  : 은닉→은닉  (hidden_size × hidden_size)  ← RNN의 핵심")
print("   Wy  : 은닉→출력  (hidden_size × output_size)")
time.sleep(0.6)
input_size  = 1           # 한 타임스텝에 수익률 1개 입력
hidden_size = 16
output_size = 1
lr          = 0.005
epochs      = 600
scale       = 0.05

Wx = np.random.randn(input_size,  hidden_size) * scale
Wh = np.random.randn(hidden_size, hidden_size) * scale
bh = np.zeros((1, hidden_size))
Wy = np.random.randn(hidden_size, output_size) * scale
by = np.zeros((1, output_size))
print(f"   → Wx{Wx.shape}  Wh{Wh.shape}  Wy{Wy.shape}")
print(f"   → 학습률={lr}  에폭={epochs}  은닉 크기={hidden_size}")
time.sleep(0.4)


# ── 헬퍼 함수 ─────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def forward_rnn(x_seq, h0):
    """x_seq: (seq_len, input_size) → 반환: y_pred scalar, 중간값 캐시"""
    h = h0.copy()
    hs, xs, zs = [], [], []
    for t in range(len(x_seq)):
        xt = x_seq[t].reshape(1, input_size)
        zt = xt @ Wx + h @ Wh + bh          # 선형 변환
        h  = np.tanh(zt)                     # 활성화 (tanh)
        hs.append(h.copy())
        xs.append(xt)
        zs.append(zt)
    y_raw  = hs[-1] @ Wy + by
    y_pred = sigmoid(y_raw)
    return y_pred, hs, xs, zs


def bptt(x_seq, y_true, hs, xs, zs):
    """BPTT: BackPropagation Through Time — 시간을 거슬러 기울기 계산"""
    global Wx, Wh, Wy, bh, by
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbh = np.zeros_like(bh)
    dWy = np.zeros_like(Wy)
    dby = np.zeros_like(by)

    y_pred, *_ = forward_rnn(x_seq, np.zeros((1, hidden_size)))
    dy = y_pred - y_true                      # BCE 미분

    dWy += hs[-1].T @ dy
    dby += dy
    dh_next = dy @ Wy.T                       # 출력층 → 은닉층

    for t in reversed(range(len(x_seq))):     # 시간 역방향
        dh = dh_next * (1 - hs[t] ** 2)      # tanh 미분
        dbh += dh
        dWx += xs[t].T @ dh
        dWh += (hs[t - 1].T @ dh if t > 0 else np.zeros((hidden_size, hidden_size)))
        dh_next = dh @ Wh.T                   # 이전 시점으로 전파

    for grad in [dWx, dWh, dWy, dbh, dby]:   # 기울기 폭발 방지
        np.clip(grad, -5, 5, out=grad)

    Wx -= lr * dWx
    Wh -= lr * dWh
    bh -= lr * dbh
    Wy -= lr * dWy
    by -= lr * dby


# ── 3. 학습 ───────────────────────────────────────────────
print("\n[3/8] BPTT 학습 시작...")
print("   순전파(→) : 시퀀스를 앞에서 뒤로 읽으며 은닉 상태 전달")
print("   역전파(←) : 오차를 뒤에서 앞으로 전파 (BPTT)")
time.sleep(0.8)

loss_history = []
prev_loss = None
for epoch in range(epochs):
    epoch_loss = 0.0
    idx = np.random.permutation(len(X_train))
    for i in idx:
        x_seq  = X_train[i]
        y_true = y_train[i]
        y_pred, hs, xs, zs = forward_rnn(x_seq, np.zeros((1, hidden_size)))
        loss = -(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        epoch_loss += loss.item()
        bptt(x_seq, y_true, hs, xs, zs)

    avg_loss = epoch_loss / len(X_train)
    loss_history.append(avg_loss)

    if epoch % 100 == 0:
        trend = ""
        if prev_loss is not None:
            trend = " ↓" if avg_loss < prev_loss else " →"
        print(f"   Epoch {epoch:4d} | Loss: {avg_loss:.4f}{trend}")
        prev_loss = avg_loss
        time.sleep(0.25)

print(f"   Epoch {epochs:4d} | 학습 완료!")
time.sleep(0.4)

# ── 4. 테스트 정확도 ───────────────────────────────────────
print("\n[4/8] 테스트 세트 정확도 평가 중...")
time.sleep(0.4)
preds, trues = [], []
for i in range(len(X_test)):
    y_pred, *_ = forward_rnn(X_test[i], np.zeros((1, hidden_size)))
    preds.append(1 if y_pred[0, 0] >= 0.5 else 0)
    trues.append(int(y_test[i, 0]))
acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
print(f"   → 테스트 정확도: {acc:.4f} ({acc * 100:.1f}%)")
print(f"   → 예측 샘플: {preds[:10]}")
print(f"   → 실제 라벨: {trues[:10]}")
time.sleep(0.4)

# ── 5. 시각화 ─────────────────────────────────────────────
print("\n[5/8] 시각화 저장 중...")
time.sleep(0.5)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 전체 그래프 요약 제목
fig.text(0.5, 0.98, '바닐라 RNN — 과거를 기억하지만 오래된 기억은 잊어버리는 한계가 있어요',
         ha='center', fontsize=9, color='#333', weight='bold')

axes[0].plot(loss_history, color='steelblue', linewidth=1.5)
axes[0].set_title("RNN 학습 손실 (BCE Loss)")
axes[0].set_xlabel("에폭")
axes[0].set_ylabel("손실")
axes[0].grid(alpha=0.3)
# 시작점 어노테이션
axes[0].annotate('처음엔 많이 틀려요',
                 xy=(0.02, 0.92), xytext=(0.15, 0.85),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='darkred')
# 수렴 어노테이션
axes[0].annotate('더 배워도 잘 안 좋아져요 (한계)',
                 xy=(0.90, 0.25), xytext=(0.50, 0.40),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# ln2 기준선
axes[0].axhline(0.693, color='orange', linewidth=0.8, linestyle='--')
axes[0].text(0.5, 0.56, '동전 던지기와 같은 수준 (ln2 ≈ 0.693)',
             transform=axes[0].transAxes, ha='center', fontsize=7, color='gray')
axes[0].text(0.5, -0.18, '손실이 낮을수록 예측이 더 정확해집니다',
             transform=axes[0].transAxes, ha='center', fontsize=7, color='gray')

colors = ['tomato' if p == 1 else 'royalblue' for p in preds[:40]]
axes[1].bar(range(40), [1] * 40, color=colors, alpha=0.8, edgecolor='k', linewidth=0.3)
match_colors = ['gold' if preds[i] == trues[i] else 'black' for i in range(40)]
for i, c in enumerate(match_colors):
    axes[1].plot(i, 0.5, 'o', color=c, markersize=4)
axes[1].set_title(f"예측 방향 (빨=상승/파=하락)  정확도={acc:.2f}")
axes[1].set_xlabel("테스트 샘플")
axes[1].set_ylabel("")
axes[1].set_yticks([])
# 정답 점 설명
axes[1].annotate('정답! (금색 점)',
                 xy=(0.05, 0.52), xytext=(0.12, 0.75),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 오답 점 설명
axes[1].annotate('틀렸어요 (검정 점)',
                 xy=(0.20, 0.52), xytext=(0.30, 0.20),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 빨간 막대 설명
axes[1].text(0.15, 0.90, '빨강: 상승 예측', transform=axes[1].transAxes,
             fontsize=7, color='tomato', ha='center')
# 파란 막대 설명
axes[1].text(0.75, 0.90, '파랑: 하락 예측', transform=axes[1].transAxes,
             fontsize=7, color='royalblue', ha='center')
# 정확도 비교 텍스트
axes[1].text(0.5, -0.18, f'정확도 {acc*100:.1f}% — 동전 던지기(50%)보다 조금 나아요',
             transform=axes[1].transAxes, ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
ticker_tag = TICKER.replace('.', '_')
plt.savefig(f"result/RnnBackprop_{ticker_tag}.png", dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: result/RnnBackprop_{ticker_tag}.png")

print("\n✓ 바닐라 RNN(BPTT) 실습 완료!\n")
