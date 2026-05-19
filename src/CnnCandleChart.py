import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 65)
print("  2D CNN: 캔들차트 이미지 → 상승/하락 패턴 분류 실습")
print("=" * 65)
print()
print("  캔들차트 → 이미지 변환 흐름:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  OHLC 주가 데이터 (시가·고가·저가·종가)               │")
print("  │       ↓  matplotlib으로 캔들 렌더링                   │")
print("  │  픽셀 이미지 (32×32 그레이스케일)                     │")
print("  │       ↓  2D CNN 입력                                  │")
print("  │  Conv2d → 이미지 패턴 감지 (양봉연속/음봉반전 등)     │")
print("  │       ↓                                               │")
print("  │  상승(1) / 하락(0) 분류                               │")
print("  └──────────────────────────────────────────────────────┘")

torch.manual_seed(42)
np.random.seed(42)

IMG_SIZE = 32   # 캔들차트 이미지 크기
N_CANDLES = 10  # 차트 1장에 표시할 캔들 수

# ── 1. OHLC 데이터 생성 ────────────────────────────────────
print("\n[1/8] 가상 OHLC 주가 데이터 생성 중...")
print("   OHLC = Open(시가) · High(고가) · Low(저가) · Close(종가)")
time.sleep(0.5)

def make_ohlc(days=600):
    """현실적인 OHLC 시계열 생성"""
    close = np.zeros(days)
    close[0] = 100.0
    for i in range(1, days):
        close[i] = close[i - 1] * (1 + np.random.normal(0, 0.015))
    open_  = close * (1 + np.random.normal(0, 0.005, days))
    high   = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.008, days)))
    low    = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.008, days)))
    return open_, high, low, close

TICKER = '078935.KS'
open_ = high = low = close = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        open_ = df['Open'].squeeze().dropna().values.flatten().astype(np.float32)
        high  = df['High'].squeeze().dropna().values.flatten().astype(np.float32)
        low   = df['Low'].squeeze().dropna().values.flatten().astype(np.float32)
        close = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        min_len = min(len(open_), len(high), len(low), len(close))
        open_, high, low, close = open_[:min_len], high[:min_len], low[:min_len], close[:min_len]
        print(f"   ✓ {TICKER}: {min_len}일 OHLC 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if close is None:
    open_, high, low, close = make_ohlc(600)
    print(f"   → 가상 600일치 OHLC 생성")

print(f"   → {len(close)}일치 OHLC  |  가격 범위: {low.min():.1f} ~ {high.max():.1f}")
time.sleep(0.4)

# ── 2. 캔들차트 → 이미지 변환 함수 ───────────────────────
print("\n[2/8] 캔들차트 → 픽셀 이미지 변환 함수 준비 중...")
print("   방법: matplotlib으로 캔들을 그린 뒤 canvas에서 픽셀 배열 추출")
time.sleep(0.5)


def candle_to_image(o, h, l, c, size=IMG_SIZE):
    """N개 캔들 → (size, size) 그레이스케일 numpy 배열"""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=size // 2)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    n = len(o)
    price_min = l.min() * 0.998
    price_max = h.max() * 1.002
    ax.set_ylim(price_min, price_max)
    ax.set_xlim(-0.5, n - 0.5)

    for i in range(n):
        color = 'red' if c[i] >= o[i] else 'blue'
        # 몸통
        body_bot = min(o[i], c[i])
        body_h   = abs(c[i] - o[i]) + 1e-6
        ax.add_patch(mpatches.Rectangle(
            (i - 0.3, body_bot), 0.6, body_h,
            color=color, linewidth=0
        ))
        # 꼬리
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.8)

    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h_px = fig.canvas.get_width_height()
    img = buf.reshape(h_px, w, 4)[:, :, :3]  # RGBA → RGB (컬러 유지)
    plt.close(fig)
    # size×size 리샘플
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img.astype(np.uint8)).resize((size, size))
    return np.array(img_pil, dtype=np.float32) / 255.0


print("   → 변환 함수 준비 완료")
time.sleep(0.3)

# ── 3. 데이터셋 생성 ───────────────────────────────────────
print("\n[3/8] 캔들 이미지 데이터셋 생성 중 (약 1~2분 소요)...")
print("   각 창마다 10개 캔들 이미지를 렌더링하고 픽셀로 변환합니다")
time.sleep(0.5)

images, labels = [], []
total = len(close) - N_CANDLES - 1
for i in range(0, total, 3):   # 3일 간격으로 샘플링
    o = open_[i:i + N_CANDLES]
    h = high[i:i + N_CANDLES]
    l = low[i:i + N_CANDLES]
    c = close[i:i + N_CANDLES]
    img = candle_to_image(o, h, l, c)
    images.append(img)
    labels.append(1 if close[i + N_CANDLES] > close[i + N_CANDLES - 1] else 0)
    if (len(images)) % 30 == 0:
        print(f"   {len(images)}/{total // 3 + 1} 이미지 생성 중...")
        time.sleep(0.05)

images = np.array(images, dtype=np.float32)  # (N, 32, 32, 3)
labels = np.array(labels, dtype=np.int64)
print(f"   → 총 {len(images)}장 생성  |  상승: {labels.sum()}장  하락: {(labels==0).sum()}장")
time.sleep(0.3)

# ── 4. Tensor 변환 & 분할 ─────────────────────────────────
print("\n[4/8] Tensor 변환 & 학습/테스트 분리 중 (8:2)...")
time.sleep(0.4)
X_tensor = torch.tensor(images).permute(0, 3, 1, 2)  # (N, 3, 32, 32)
y_tensor = torch.tensor(labels)
split = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]
print(f"   → 학습: {X_train.shape}  테스트: {X_test.shape}")
time.sleep(0.3)

# ── 5. 2D CNN 모델 ─────────────────────────────────────────
print("\n[5/8] 2D CNN 모델 정의 중...")
print("   Conv2d(3→16, 3×3)  : RGB 3채널 이미지의 2D 패턴(캔들 형태) 감지")
print("   Conv2d(16→32, 3×3) : 더 복잡한 복합 패턴 감지")
print("   GlobalAvgPool → FC : 위치 무관하게 패턴 집계 → 분류")
time.sleep(0.5)


class CNN2DCandle(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 32→16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 16→8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                         # 고정 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model2d = CNN2DCandle()
total_params = sum(p.numel() for p in model2d.parameters())
print(f"   → 총 파라미터: {total_params:,}개")
time.sleep(0.4)

# ── 6. 학습 ───────────────────────────────────────────────
print("\n[6/8] 2D CNN 학습 시작...")
time.sleep(0.6)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model2d.parameters(), lr=0.001)
EPOCHS, BATCH = 80, 32
loss_hist, acc_hist = [], []
prev_loss = None
model2d.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    ep_loss, ep_correct = 0.0, 0
    for s in range(0, len(X_train), BATCH):
        idx = perm[s:s + BATCH]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        out = model2d(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        ep_loss    += loss.item() * len(xb)
        ep_correct += (out.argmax(1) == yb).sum().item()
    avg_loss = ep_loss / len(X_train)
    avg_acc  = ep_correct / len(X_train)
    loss_hist.append(avg_loss)
    acc_hist.append(avg_acc)
    if epoch % 20 == 0:
        trend = " ↓" if (prev_loss and avg_loss < prev_loss) else ""
        print(f"   Epoch {epoch:3d} | Loss: {avg_loss:.4f}{trend}  Acc: {avg_acc:.4f}")
        prev_loss = avg_loss
        time.sleep(0.2)
print(f"   Epoch {EPOCHS:3d} | 학습 완료!")
time.sleep(0.4)

# ── 7. 테스트 평가 ─────────────────────────────────────────
print("\n[7/8] 테스트 평가 중...")
time.sleep(0.4)
model2d.eval()
with torch.no_grad():
    out_test = model2d(X_test)
    acc_test = (out_test.argmax(1) == y_test).float().mean().item()
print(f"   → 테스트 정확도: {acc_test:.4f} ({acc_test * 100:.1f}%)")
time.sleep(0.3)

# ── 8. 시각화 ─────────────────────────────────────────────
print("\n[8/8] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(14, 10))

# 샘플 캔들 이미지 6장
for i in range(6):
    ax = fig.add_subplot(3, 4, i + 1)
    ax.imshow(images[i * 20])
    label_str = "상승▲" if labels[i * 20] == 1 else "하락▼"
    color = 'red' if labels[i * 20] == 1 else 'blue'
    ax.set_title(f"샘플{i + 1}: {label_str}", color=color, fontsize=9)
    ax.axis('off')

# 학습 손실
ax_loss = fig.add_subplot(3, 2, 5)
ax_loss.plot(loss_hist, color='steelblue')
ax_loss.set_title("학습 손실")
ax_loss.set_xlabel("에폭")
ax_loss.set_ylabel("손실")
ax_loss.grid(alpha=0.3)

# 학습 정확도
ax_acc = fig.add_subplot(3, 2, 6)
ax_acc.plot(acc_hist, color='tomato')
ax_acc.axhline(0.5, linestyle='--', color='gray', linewidth=0.8, label='무작위=0.5')
ax_acc.set_title(f"학습 정확도  |  테스트 정확도={acc_test:.2f}")
ax_acc.set_xlabel("에폭")
ax_acc.set_ylabel("정확도")
ax_acc.legend()
ax_acc.grid(alpha=0.3)

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         "2D CNN — 캔들차트를 '사진'으로 보고 상승/하락 패턴을 인식합니다",
         ha='center', fontsize=9, color='#333', weight='bold')

# 캔들 이미지 전체 설명 (첫 번째 이미지 위)
fig.text(0.5, 0.68,
         "캔들차트를 32×32 픽셀 사진으로 변환해 AI에게 보여줍니다",
         ha='center', fontsize=8, color='#333')

# 첫 번째 이미지(샘플1) 아래 설명
ax_first = fig.axes[0]
ax_first.text(0.5, -0.18,
              '빨간 막대=오른 날(양봉), 파란 막대=내린 날(음봉)',
              transform=ax_first.transAxes, ha='center', fontsize=7, color='gray')

# 손실 패널 설명
ax_loss.text(0.5, -0.22,
             '아래로 내려갈수록 이미지 패턴을 더 잘 인식하는 중',
             transform=ax_loss.transAxes, ha='center', fontsize=7, color='gray')

# 정확도 패널 설명
ax_acc.text(0.5, -0.22,
            '위로 올라갈수록 상승/하락 예측을 더 잘 맞혀요',
            transform=ax_acc.transAxes, ha='center', fontsize=7, color='gray')

plt.subplots_adjust(top=0.93)
plt.tight_layout()
ticker_tag = TICKER.replace(".", "_")
out_name = f"../result/CnnCandleChart_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ 2D CNN 캔들차트 이미지 분류 실습 완료!\n")
