"""
시계열 윈도우 생성기 — 사용자 입력으로 윈도우/예측기간/종목/스텝 설정
실행 방법:
  python3 TimeSeriesWindow.py                     # 기본 대화형 입력
  python3 TimeSeriesWindow.py --ticker AAPL --window 30 --horizon 5 --step 1
"""
import os
import sys
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

# ── 파라미터 파싱 ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="시계열 윈도우 기반 주가 예측")
parser.add_argument('--ticker',  type=str,   default=None,
                    help="종목 티커 (예: AAPL, 005930.KS, ^GSPC). 없으면 가상 데이터")
parser.add_argument('--window',  type=int,   default=None,
                    help="입력 윈도우 크기 (과거 몇 일치?). 기본=30")
parser.add_argument('--horizon', type=int,   default=None,
                    help="예측 기간 (며칠 후?). 기본=1")
parser.add_argument('--step',    type=int,   default=None,
                    help="윈도우 이동 스텝. 기본=1")
args = parser.parse_args()

print("=" * 65)
print("  시계열 윈도우 생성 & 주가 예측 실습")
print("=" * 65)
print()
print("  핵심 개념:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  윈도우 크기   : 모델이 '기억'하는 과거 날짜 수      │")
print("  │  예측 기간     : 며칠 후 가격을 맞출지               │")
print("  │  스텝 크기     : 윈도우를 몇 칸씩 이동할지           │")
print("  │  예) window=30, horizon=5, step=1                    │")
print("  │  → 과거 30일로 5일 후 가격을 1일씩 이동하며 학습     │")
print("  └──────────────────────────────────────────────────────┘")

# ── 대화형 입력 (argparse 값 없을 때) ──────────────────────
def ask_int(prompt, default, min_val, max_val):
    if sys.stdin.isatty():
        while True:
            raw = input(f"{prompt} (기본={default}, {min_val}~{max_val}): ").strip()
            if raw == "":
                return default
            try:
                val = int(raw)
                if min_val <= val <= max_val:
                    return val
                print(f"   {min_val}~{max_val} 사이 값을 입력하세요.")
            except ValueError:
                print("   숫자를 입력하세요.")
    return default

def ask_str(prompt, default):
    if sys.stdin.isatty():
        raw = input(f"{prompt} (기본={default}, 엔터=건너뜀): ").strip()
        return raw if raw else default
    return default

print()
# 기본값: GS피앤엘 (078935.KS), 윈도우 20일, 예측 5일, 스텝 1
TICKER  = args.ticker  if args.ticker  is not None else ask_str("   종목 티커", "078935.KS")
WINDOW  = args.window  if args.window  is not None else ask_int("   윈도우 크기 (일)", 20, 5, 120)
HORIZON = args.horizon if args.horizon is not None else ask_int("   예측 기간 (일)", 5, 1, 30)
STEP    = args.step    if args.step    is not None else ask_int("   스텝 크기", 1, 1, 10)

print(f"\n   설정 → 종목={TICKER}  윈도우={WINDOW}일  예측={HORIZON}일 후  스텝={STEP}")
time.sleep(0.3)

# ── 1. 데이터 로드 ─────────────────────────────────────────
print("\n[1/7] 주가 데이터 로드 중...")
time.sleep(0.4)

prices = None
try:
    import yfinance as yf
    from datetime import date
    today = date.today().isoformat()  # yfinance가 마지막 거래일까지 자동 반환
    df = yf.download(TICKER, start='2020-01-01', end=today,
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
    else:
        print(f"   ✗ {TICKER} 데이터 부족 → 가상 데이터 사용")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    np.random.seed(42)
    n = 1000
    t = np.arange(n, dtype=float)
    prices = (150 + 0.1*t + 12*np.sin(t/40) + 5*np.sin(t/10)
              + np.random.normal(0, 2.5, n)).astype(np.float32)
    print(f"   → 가상 주가 {len(prices)}일치 생성")

print(f"   가격 범위: {prices.min():.2f} ~ {prices.max():.2f}")
time.sleep(0.3)

# ── 2. Min-Max 정규화 ─────────────────────────────────────
print("\n[2/7] Min-Max 정규화 중...")
time.sleep(0.3)
p_min, p_max = prices.min(), prices.max()
norm = (prices - p_min) / (p_max - p_min + 1e-8)

# ── 3. 윈도우 데이터셋 생성 ────────────────────────────────
print("\n[3/7] 슬라이딩 윈도우 데이터셋 생성 중...")
print(f"   윈도우={WINDOW}일  예측={HORIZON}일 후  스텝={STEP}")
time.sleep(0.4)

X_list, y_list, idx_list = [], [], []
i = 0
while i + WINDOW + HORIZON <= len(norm):
    X_list.append(norm[i:i + WINDOW])
    y_list.append(norm[i + WINDOW + HORIZON - 1])
    idx_list.append(i)
    i += STEP

X_np = np.array(X_list, dtype=np.float32)   # (N, WINDOW)
y_np = np.array(y_list, dtype=np.float32)   # (N,)
print(f"   → 총 {len(X_np)}개 윈도우 생성 (스텝={STEP})")
time.sleep(0.3)

# ── 4. PyTorch LSTM 예측 모델 ─────────────────────────────
print("\n[4/7] LSTM 예측 모델 학습 중...")
time.sleep(0.4)
import torch
import torch.nn as nn

torch.manual_seed(42)
split = int(len(X_np) * 0.8)
X_tr = torch.tensor(X_np[:split]).unsqueeze(-1)    # (N, window, 1)
y_tr = torch.tensor(y_np[:split]).unsqueeze(-1)    # (N, 1)
X_te = torch.tensor(X_np[split:]).unsqueeze(-1)
y_te = torch.tensor(y_np[split:]).unsqueeze(-1)
print(f"   학습: {X_tr.shape}  테스트: {X_te.shape}")


class WindowLSTM(nn.Module):
    def __init__(self, win, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True,
                            dropout=0.2)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


model = WindowLSTM(WINDOW)
crit  = nn.MSELoss()
opt   = torch.optim.Adam(model.parameters(), lr=0.001)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=80, gamma=0.5)
EPOCHS, BATCH = 160, 32
loss_hist = []
model.train()
for epoch in range(EPOCHS):
    perm   = torch.randperm(len(X_tr))
    ep_loss = 0.0
    for s in range(0, len(X_tr), BATCH):
        idx = perm[s:s + BATCH]
        xb, yb = X_tr[idx], y_tr[idx]
        opt.zero_grad()
        pred = model(xb)
        loss = crit(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item() * len(xb)
    sched.step()
    avg = ep_loss / len(X_tr)
    loss_hist.append(avg)
    if epoch % 40 == 0:
        print(f"   Epoch {epoch:4d} | Loss: {avg:.6f}")
        time.sleep(0.1)
print(f"   Epoch {EPOCHS:4d} | 학습 완료!")
time.sleep(0.3)

# ── 5. 테스트 평가 ─────────────────────────────────────────
print("\n[5/7] 테스트 평가 중...")
time.sleep(0.3)
model.eval()
with torch.no_grad():
    pred_norm = model(X_te).numpy().flatten()
    true_norm = y_te.numpy().flatten()
pred_real = pred_norm * (p_max - p_min) + p_min
true_real = true_norm * (p_max - p_min) + p_min
mae  = np.mean(np.abs(pred_real - true_real))
rmse = np.sqrt(np.mean((pred_real - true_real)**2))
print(f"   MAE={mae:.4f}  RMSE={rmse:.4f}")
time.sleep(0.3)

# ── 6. 미래 예측 (horizon일 후) ───────────────────────────
print(f"\n[6/7] 최근 {WINDOW}일로 {HORIZON}일 후 가격 예측 중...")
time.sleep(0.3)
last_win = torch.tensor(norm[-WINDOW:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
with torch.no_grad():
    future_norm = model(last_win).item()
future_price = future_norm * (p_max - p_min) + p_min
print(f"   현재 가격:       {prices[-1]:.2f}")
print(f"   {HORIZON}일 후 예측: {future_price:.2f}  "
      f"({'▲ 상승' if future_price > prices[-1] else '▼ 하락'} 예상)")
time.sleep(0.3)

# ── 7. 시각화 ─────────────────────────────────────────────
print("\n[7/7] 시각화 저장 중...")
time.sleep(0.5)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 윈도우 샘플 시각화
ax = axes[0, 0]
for k in range(min(5, len(X_np))):
    offset = k * (WINDOW // 5)
    ax.plot(X_np[k * (len(X_np) // 5)], alpha=0.7, linewidth=1.2,
            label=f"윈도우{k+1}")
ax.set_title(f"윈도우 샘플 5개 (크기={WINDOW}일, 정규화됨)")
ax.set_xlabel("시퀀스 위치")
ax.set_ylabel("정규화 가격")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 학습 손실
ax = axes[0, 1]
ax.plot(loss_hist, color='steelblue')
ax.set_title("LSTM 학습 손실 (MSE)")
ax.set_xlabel("에폭")
ax.set_ylabel("MSE 손실")
ax.grid(alpha=0.3)

# 예측 vs 실제
ax = axes[1, 0]
n_show = min(100, len(true_real))
ax.plot(true_real[:n_show], color='steelblue', linewidth=1.5, label='실제 가격')
ax.plot(pred_real[:n_show], '--', color='tomato', linewidth=1.5, label=f'{HORIZON}일 후 예측')
ax.set_title(f"예측 vs 실제  |  MAE={mae:.2f}  RMSE={rmse:.2f}")
ax.set_xlabel("테스트 인덱스")
ax.set_ylabel("가격")
ax.legend()
ax.grid(alpha=0.3)

# 설정 요약 텍스트
ax = axes[1, 1]
ax.axis('off')
summary = (
    "실행 설정 요약\n\n"
    f"• 종목:       {TICKER}\n"
    f"• 윈도우 크기: {WINDOW}일\n"
    f"• 예측 기간:  {HORIZON}일 후\n"
    f"• 스텝 크기:  {STEP}\n"
    f"• 총 샘플:    {len(X_np)}개\n"
    f"• 학습 샘플:  {len(X_tr)}개\n"
    f"• 테스트 샘플:{len(X_te)}개\n\n"
    "예측 결과\n\n"
    f"• MAE:  {mae:.4f}\n"
    f"• RMSE: {rmse:.4f}\n"
    f"• 현재: {prices[-1]:.2f}\n"
    f"• 예측: {future_price:.2f} "
    f"({'▲' if future_price > prices[-1] else '▼'})\n\n"
    "CLI 사용법:\n"
    "python3 TimeSeriesWindow.py \\\n"
    "  --ticker AAPL \\\n"
    "  --window 30 \\\n"
    "  --horizon 5 \\\n"
    "  --step 1"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         f"최근 {WINDOW}일 주가 모양을 보고 {HORIZON}일 후 가격을 예측합니다",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 윈도우 샘플 패널 ──
ax_win = axes[0, 0]
ax_win.text(0.5, 1.04,
            '각 선 = 과거 20일치 주가 모양',
            transform=ax_win.transAxes, ha='center', fontsize=8, color='#333')
ax_win.text(0.5, -0.16,
            f'이 모양을 보고 {HORIZON}일 후를 예측해요',
            transform=ax_win.transAxes, ha='center', fontsize=7, color='gray')

# ── 손실 패널 ──
ax_loss2 = axes[0, 1]
ax_loss2.annotate('처음엔 많이 틀려요', xy=(0.0, 0.95), xytext=(0.08, 0.82),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax_loss2.annotate('이제 잘 배웠어요', xy=(0.92, 0.12), xytext=(0.60, 0.32),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')

# ── 예측 vs 실제 패널 ──
ax_pred = axes[1, 0]
ax_pred.text(0.5, -0.16,
             '파란선=실제, 점선=예측',
             transform=ax_pred.transAxes, ha='center', fontsize=7, color='gray')
ax_pred.annotate('잘 맞춰요!', xy=(0.25, 0.55), xytext=(0.10, 0.75),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='steelblue'), fontsize=7, color='steelblue')
ax_pred.annotate('급변은 어려워요', xy=(0.70, 0.35), xytext=(0.50, 0.15),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=7, color='tomato')

plt.subplots_adjust(top=0.93)
plt.tight_layout()
out_name = f"../result/TimeSeriesWindow_{TICKER.replace('.','_')}_w{WINDOW}_h{HORIZON}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print(f"\n✓ 시계열 윈도우 주가 예측 실습 완료! (종목={TICKER}, 윈도우={WINDOW}, 예측={HORIZON}일)\n")
