import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("result", exist_ok=True)

print("=" * 65)
print("  CNN+LSTM 하이브리드: 로컬 패턴 추출 + 장기 기억 분류 실습")
print("=" * 65)
print()
print("  하이브리드 구조:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  입력: (배치, 윈도우 수, 윈도우 크기)                 │")
print("  │  ① Conv1d  : 각 짧은 윈도우 내 로컬 패턴 감지        │")
print("  │  ② LSTM    : 여러 윈도우에 걸친 시간 순서 기억        │")
print("  │  ③ Linear  : 상승/하락 최종 분류                      │")
print("  │                                                       │")
print("  │  비유: CNN = 하루 뉴스 요약, LSTM = 주간 흐름 파악    │")
print("  └──────────────────────────────────────────────────────┘")

torch.manual_seed(42)
np.random.seed(42)

# ── 1. 데이터 생성 ─────────────────────────────────────────
print("\n[1/8] 주가 시계열 로드 중 (078935.KS - GS피앤엘)...")
time.sleep(0.5)

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
    days = 600
    t = np.arange(days, dtype=float)
    prices = (100 + 0.1 * t + 10 * np.sin(t / 30) + 5 * np.sin(t / 7) + np.random.normal(0, 2.0, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
returns = np.diff(prices) / prices[:-1]
print(f"   → {days}일치 주가  |  수익률 범위: {returns.min():.3f} ~ {returns.max():.3f}")
time.sleep(0.3)

# ── 2. 하이브리드 입력 구성: 윈도우 분할 ──────────────────
print("\n[2/8] 하이브리드 입력 구성 중...")
print("   전략: 60일 시퀀스를 6개×10일 윈도우로 분할")
print("         CNN → 각 10일 윈도우의 로컬 패턴 추출")
print("         LSTM → 6개 패턴 벡터의 시간 순서 학습")
time.sleep(0.5)

N_WINDOWS  = 6   # LSTM 시퀀스 길이 (타임스텝 수)
WIN_SIZE   = 10  # 각 CNN 윈도우 길이
TOTAL_LEN  = N_WINDOWS * WIN_SIZE  # = 60일

X_list, y_list = [], []
for i in range(len(returns) - TOTAL_LEN):
    seq = returns[i:i + TOTAL_LEN]                         # (60,)
    # 윈도우별 z-score 정규화: 패턴을 더 선명하게
    mu, sigma = seq.mean(), seq.std() + 1e-8
    seq = (seq - mu) / sigma
    windows = seq.reshape(N_WINDOWS, WIN_SIZE)             # (6, 10)
    X_list.append(windows)
    y_list.append(1 if returns[i + TOTAL_LEN] > 0 else 0)

X_np = np.array(X_list, dtype=np.float32)   # (N, 6, 10)
y_np = np.array(y_list, dtype=np.int64)
print(f"   → 샘플 수: {len(X_np)}  |  상승: {y_np.sum()}  하락: {(y_np==0).sum()}")
time.sleep(0.3)

# ── 3. Tensor 변환 & 분할 ─────────────────────────────────
print("\n[3/8] Tensor 변환 & 학습/테스트 분리 중 (8:2)...")
time.sleep(0.4)
X_tensor = torch.tensor(X_np)          # (N, 6, 10)
y_tensor = torch.tensor(y_np)
split = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]
print(f"   → 학습: {X_train.shape}  테스트: {X_test.shape}")
time.sleep(0.3)

# ── 4. CNN+LSTM 하이브리드 모델 ────────────────────────────
print("\n[4/8] CNN+LSTM 하이브리드 모델 정의 중...")
print("   CNN 인코더  : (배치×윈도우수, 1, 10) → 각 윈도우를 특징 벡터로")
print("   LSTM 인코더 : (배치, 6, CNN출력차원) → 시퀀스 전체 패턴 기억")
print("   Linear      : LSTM 마지막 은닉 → 상승/하락")
time.sleep(0.6)


class CnnLstmHybrid(nn.Module):
    def __init__(self, win_size=10, n_windows=6, cnn_out=32, lstm_hidden=64, num_layers=2):
        super().__init__()
        self.n_windows = n_windows

        # CNN 인코더: 각 윈도우(길이=win_size)의 로컬 패턴 추출
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # LSTM: CNN이 추출한 n_windows개 특징 벡터를 순서대로 처리
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(lstm_hidden, 2)

    def forward(self, x):
        # x: (B, n_windows, win_size)
        B, N, W = x.shape

        # CNN: 윈도우별 로컬 패턴 추출
        x_cnn = x.view(B * N, 1, W)           # (B*N, 1, win_size)
        feat  = self.cnn(x_cnn)               # (B*N, cnn_out, 1)
        feat  = feat.squeeze(-1)              # (B*N, cnn_out)
        feat  = feat.view(B, N, -1)           # (B, N, cnn_out) = LSTM 입력

        # LSTM: 시간 순서로 패턴 시퀀스 학습
        out, _ = self.lstm(feat)              # (B, N, lstm_hidden)
        out = self.dropout(out[:, -1, :])     # 마지막 타임스텝
        return self.classifier(out)           # (B, 2)


model = CnnLstmHybrid(WIN_SIZE, N_WINDOWS, cnn_out=32, lstm_hidden=64)
total_params = sum(p.numel() for p in model.parameters())
print(f"   → 총 파라미터: {total_params:,}개")
time.sleep(0.4)

# ── 5. 학습 설정 ───────────────────────────────────────────
print("\n[5/8] 학습 설정 중 (LabelSmoothing + Adam + 학습률 스케줄러)...")
time.sleep(0.4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 과자신감 방지
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
EPOCHS, BATCH = 60, 32
print(f"   → 에폭={EPOCHS}  배치={BATCH}  손실=CrossEntropy+LabelSmoothing(0.1)")
time.sleep(0.3)

# ── 6. 학습 루프 ──────────────────────────────────────────
print("\n[6/8] CNN+LSTM 하이브리드 학습 시작...")
print("   CNN이 단기 패턴을 벡터로 압축 → LSTM이 그 흐름을 기억합니다")
time.sleep(0.8)

loss_hist, acc_hist = [], []
prev_loss = None
model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    ep_loss, ep_correct = 0.0, 0
    for s in range(0, len(X_train), BATCH):
        idx = perm[s:s + BATCH]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss    += loss.item() * len(xb)
        ep_correct += (out.argmax(1) == yb).sum().item()
    scheduler.step()
    avg_loss = ep_loss / len(X_train)
    avg_acc  = ep_correct / len(X_train)
    loss_hist.append(avg_loss)
    acc_hist.append(avg_acc)
    if epoch % 15 == 0:
        trend = " ↓" if (prev_loss and avg_loss < prev_loss) else ""
        print(f"   Epoch {epoch:4d} | Loss: {avg_loss:.4f}{trend}  Acc: {avg_acc:.4f}")
        prev_loss = avg_loss
        time.sleep(0.2)

print(f"   Epoch  {EPOCHS:3d} | 학습 완료!")
time.sleep(0.4)

# ── 7. 테스트 평가 ─────────────────────────────────────────
print("\n[7/8] 테스트 세트 평가 중...")
time.sleep(0.4)
model.eval()
with torch.no_grad():
    out_test = model(X_test)
    preds    = out_test.argmax(1)
    acc_test = (preds == y_test).float().mean().item()
    probs    = torch.softmax(out_test, dim=1)[:, 1].numpy()
print(f"   → 테스트 정확도: {acc_test:.4f} ({acc_test * 100:.1f}%)")
time.sleep(0.3)

# ── 8. 시각화 ─────────────────────────────────────────────
print("\n[8/8] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(14, 10))

# 학습 손실
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(loss_hist, color='steelblue', linewidth=1.5)
ax1.set_title("학습 손실 (CrossEntropy)")
ax1.set_xlabel("에폭")
ax1.set_ylabel("손실")
ax1.grid(alpha=0.3)

# 학습 정확도
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(acc_hist, color='tomato', linewidth=1.5)
ax2.axhline(0.5, linestyle='--', color='gray', linewidth=0.8, label='무작위=0.5')
ax2.set_title(f"학습 정확도  |  테스트={acc_test:.2f}")
ax2.set_xlabel("에폭")
ax2.set_ylabel("정확도")
ax2.legend()
ax2.grid(alpha=0.3)

# 테스트 예측 확률 (중앙값 기준으로 색 분류 — 항상 빨강/파랑 섞임)
ax3 = fig.add_subplot(2, 1, 2)
n_show = min(80, len(probs))
prob_median = float(np.median(probs[:n_show]))
colors = ['tomato' if p >= prob_median else 'royalblue' for p in probs[:n_show]]
ax3.bar(range(n_show), probs[:n_show], color=colors, alpha=0.8, edgecolor='k', linewidth=0.2)
ax3.axhline(0.5, linestyle=':', color='gray', linewidth=0.8, label='0.5 기준선')
ax3.axhline(prob_median, linestyle='--', color='black', linewidth=1.0,
            label=f'중앙값={prob_median:.3f} (색 기준)')
ax3.legend(fontsize=7)
ax3.set_title(f"테스트 상승 예측 확률 — 빨강=중앙값 이상 / 파랑=중앙값 미만 (정확도={acc_test:.2f})")
ax3.set_xlabel("테스트 샘플")
ax3.set_ylabel("상승 확률")

# 모델 구조 텍스트
ax_txt = fig.add_axes([0.01, 0.52, 0.22, 0.44])
ax_txt.axis('off')
struct = (
    "CNN+LSTM 구조\n\n"
    f"입력: z-score 정규화\n"
    f"(배치, {N_WINDOWS}, {WIN_SIZE})\n\n"
    "① Conv1d(1→16, k=3)\n"
    "   ReLU\n"
    "② Conv1d(16→32, k=3)\n"
    "   ReLU\n"
    "③ AdaptiveAvgPool1d(1)\n"
    "   → 윈도우별 32-dim 벡터\n\n"
    f"④ LSTM(32→64)\n"
    f"   num_layers=2\n"
    f"   Dropout=0.3\n\n"
    f"⑤ Linear(64→2)\n"
    f"   LabelSmoothing=0.1\n\n"
    f"파라미터: {total_params:,}개"
)
ax_txt.text(0.05, 0.95, struct, transform=ax_txt.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         "CNN(단기 패턴 포착) + LSTM(장기 흐름 기억) — 두 장점을 합쳤어요",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 손실 패널 (ax1) ──
ax1.annotate('처음엔 많이 틀려요', xy=(0.0, 0.95), xytext=(0.08, 0.82),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax1.annotate('이제 잘 배웠어요', xy=(0.9, 0.25), xytext=(0.60, 0.45),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')

# ── 정확도 패널 (ax2) ──
ax2.text(0.5, -0.18,
         '이 아래면 동전 던지기보다 못해요 (점선=0.5)',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')

# ── 예측확률 패널 (ax3) ──
ax3.text(0.10, 0.88, '상대적 상승 예측\n(중앙값 이상)', transform=ax3.transAxes,
         fontsize=7, color='tomato', ha='center')
ax3.text(0.80, 0.15, '상대적 하락 예측\n(중앙값 미만)', transform=ax3.transAxes,
         fontsize=7, color='royalblue', ha='center')

# ── 구조 텍스트 박스 위 설명 ──
ax_txt.text(0.05, 1.04,
            'CNN이 단기 패턴을 요약 → LSTM이 흐름을 기억',
            transform=ax_txt.transAxes, fontsize=8, color='#333', weight='bold')

plt.subplots_adjust(top=0.93)
plt.tight_layout()
ticker_tag = TICKER.replace(".", "_")
out_name = f"result/CnnLstmHybrid_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ CNN+LSTM 하이브리드 주가 방향 분류 실습 완료!\n")
