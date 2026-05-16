import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("result", exist_ok=True)

print("=" * 65)
print("  1D CNN: 주가 시계열 특징 자동 추출 & 방향 분류 실습")
print("=" * 65)
print()
print("  1D CNN 구조:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  입력: (배치, 채널=1, 시퀀스 길이)                    │")
print("  │  Conv1d → 시퀀스에서 로컬 패턴(필터) 감지             │")
print("  │  예) 필터 크기 3 = 연속 3일 주가 패턴을 감지          │")
print("  │  MaxPool1d → 가장 강한 패턴 신호만 남기고 압축        │")
print("  │  Flatten → 1D 벡터로 펼침                            │")
print("  │  Linear → 최종 예측(상승/하락)                        │")
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
    days = 500
    t = np.arange(days, dtype=float)
    prices = (100 + 0.1 * t + 8 * np.sin(t / 25) + np.random.normal(0, 2.0, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
returns = np.diff(prices) / prices[:-1]
print(f"   → {days}일치 주가  |  일간 수익률 범위: {returns.min():.3f} ~ {returns.max():.3f}")
time.sleep(0.3)

# ── 2. 슬라이딩 윈도우 → 분류 레이블 ──────────────────────
print("\n[2/8] 슬라이딩 윈도우 데이터셋 구성 중...")
print("   입력: 과거 30일 수익률 시퀀스")
print("   정답: 다음 날 수익률 > 0 이면 상승(1), 아니면 하락(0)")
time.sleep(0.5)
SEQ_LEN = 30
X_list, y_list = [], []
for i in range(len(returns) - SEQ_LEN):
    seq = returns[i:i + SEQ_LEN]
    mu, sigma = seq.mean(), seq.std() + 1e-8
    seq = (seq - mu) / sigma          # 윈도우별 z-score 정규화
    X_list.append(seq)
    y_list.append(1 if returns[i + SEQ_LEN] > 0 else 0)

X_all = np.array(X_list, dtype=np.float32)  # (N, 30)
y_all = np.array(y_list, dtype=np.long)

# 1D CNN 입력 형태: (N, 채널, 시퀀스) = (N, 1, 30)
X_tensor = torch.tensor(X_all).unsqueeze(1)
y_tensor = torch.tensor(y_all)

split = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]
print(f"   → 학습: {X_train.shape}  테스트: {X_test.shape}")
time.sleep(0.3)

# ── 3. 1D CNN 모델 정의 ────────────────────────────────────
print("\n[3/8] 1D CNN 모델 정의 중...")
print("   Conv1d(1→16, kernel=3)  : 3일 패턴 16종 감지")
print("   Conv1d(16→32, kernel=3) : 더 복잡한 패턴 32종 감지")
print("   MaxPool1d(2)            : 가장 강한 신호만 선택·압축")
print("   Flatten → Linear        : 최종 상승/하락 분류")
time.sleep(0.6)


class CNN1DStock(nn.Module):
    def __init__(self, seq_len=30):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # (N,16,30)
            nn.ReLU(),
            nn.MaxPool1d(2),                              # (N,16,15)
            nn.Conv1d(16, 32, kernel_size=3, padding=1), # (N,32,15)
            nn.ReLU(),
            nn.MaxPool1d(2),                              # (N,32,7)
            nn.Conv1d(32, 64, kernel_size=3, padding=1), # (N,64,7)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),                      # (N,64,4) 고정
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        feat = self.conv_block(x)
        return self.classifier(feat)

    def extract_features(self, x):
        """Conv 블록까지만 실행해 중간 특징 벡터 반환"""
        with torch.no_grad():
            return self.conv_block(x)


model = CNN1DStock(SEQ_LEN)
total_params = sum(p.numel() for p in model.parameters())
print(f"   → 총 파라미터: {total_params:,}개")
print(model)
time.sleep(0.5)

# ── 4. 학습 설정 ───────────────────────────────────────────
print("\n[4/8] 학습 설정 중 (CrossEntropy + Adam)...")
time.sleep(0.4)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
EPOCHS     = 60
BATCH_SIZE = 64
print(f"   → 에폭={EPOCHS}  배치={BATCH_SIZE}  손실=CrossEntropy+LabelSmoothing")
time.sleep(0.3)

# ── 5. 학습 ───────────────────────────────────────────────
print("\n[5/8] 1D CNN 학습 시작...")
print("   Conv 필터가 '3일 연속 상승', '역헤드앤숄더' 같은 패턴을 자동으로 찾습니다")
time.sleep(0.8)

loss_hist, acc_hist = [], []
prev_loss = None
model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    ep_loss, ep_correct = 0.0, 0
    for s in range(0, len(X_train), BATCH_SIZE):
        idx = perm[s:s + BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
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

print(f"   Epoch {EPOCHS:4d} | 학습 완료!")
time.sleep(0.4)

# ── 6. 테스트 평가 ─────────────────────────────────────────
print("\n[6/8] 테스트 세트 평가 중...")
time.sleep(0.4)
model.eval()
with torch.no_grad():
    out_test = model(X_test)
    preds    = out_test.argmax(1)
    acc_test = (preds == y_test).float().mean().item()
    probs    = torch.softmax(out_test, dim=1)[:, 1].numpy()
print(f"   → 테스트 정확도: {acc_test:.4f} ({acc_test * 100:.1f}%)")
time.sleep(0.3)

# ── 7. 특징 맵 시각화 ─────────────────────────────────────
print("\n[7/8] Conv 필터가 추출한 특징 맵 시각화 준비 중...")
print("   → 어떤 날짜 구간에서 필터가 강하게 반응했는지 확인합니다")
time.sleep(0.5)

# 테스트 샘플 1개로 첫 번째 Conv 출력 확인
sample_input = X_test[:1]  # (1, 1, 30)
hook_output  = {}

def hook_fn(module, inp, out):
    hook_output['conv1'] = out.detach()

hook = model.conv_block[0].register_forward_hook(hook_fn)
with torch.no_grad():
    model(sample_input)
hook.remove()
feat_map = hook_output['conv1'][0].numpy()  # (16채널, 30)
time.sleep(0.3)

# ── 8. 시각화 ─────────────────────────────────────────────
print("\n[8/8] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(14, 12))

# 학습 곡선
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(loss_hist, color='steelblue')
ax1.set_title("학습 손실 (CrossEntropy)")
ax1.set_xlabel("에폭")
ax1.set_ylabel("손실")
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(acc_hist, color='tomato')
ax2.axhline(0.5, linestyle='--', color='gray', linewidth=0.8, label='무작위=0.5')
ax2.set_title("학습 정확도")
ax2.set_xlabel("에폭")
ax2.set_ylabel("정확도")
ax2.legend()
ax2.grid(alpha=0.3)

# 테스트 예측 확률 (중앙값 기준으로 색 분류)
ax3 = fig.add_subplot(3, 1, 2)
n_show = min(60, len(probs))
prob_median = float(np.median(probs[:n_show]))
colors = ['tomato' if p >= prob_median else 'royalblue' for p in probs[:n_show]]
ax3.bar(range(n_show), probs[:n_show], color=colors, alpha=0.8, edgecolor='k', linewidth=0.2)
ax3.axhline(0.5, linestyle=':', color='gray', linewidth=0.8, label='0.5 기준선')
ax3.axhline(prob_median, linestyle='--', color='black', linewidth=1.0,
            label=f'중앙값={prob_median:.3f} (색 기준)')
ax3.legend(fontsize=7)
ax3.set_title(f"테스트 샘플 상승 예측 확률 — 빨강=중앙값 이상 / 파랑=중앙값 미만 (정확도={acc_test:.2f})")
ax3.set_xlabel("테스트 샘플")
ax3.set_ylabel("상승 확률")

# 특징 맵 (첫 번째 Conv 레이어의 첫 8개 필터)
ax4 = fig.add_subplot(3, 1, 3)
for i in range(min(8, feat_map.shape[0])):
    ax4.plot(feat_map[i], alpha=0.7, linewidth=1.2, label=f"필터{i + 1}")
ax4.set_title("첫 번째 Conv1d 레이어의 특징 맵 (필터별 반응 강도)")
ax4.set_xlabel("시퀀스 위치 (날짜)")
ax4.set_ylabel("활성화 값")
ax4.legend(fontsize=7, ncol=4)
ax4.grid(alpha=0.3)

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         "1D CNN — 연속된 주가에서 '3일 연속 상승' 같은 패턴을 자동으로 찾아요",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 손실 패널 (ax1) ──
ax1.annotate('처음엔 틀려요', xy=(0.0, 0.95), xytext=(0.08, 0.85),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax1.annotate('배웠어요!', xy=(0.95, 0.15), xytext=(0.70, 0.35),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')

# ── 정확도 패널 (ax2) ──
ax2.text(0.5, -0.18, '0.5 기준: 동전 던지기 수준 — 이 위로 올라가야 의미 있어요',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')
ax2.annotate('이 수준이면 좋아요', xy=(0.9, 0.85), xytext=(0.55, 0.75),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')

# ── 예측확률 패널 (ax3) ──
ax3.text(0.10, 0.88, '상대적 상승 예측\n(중앙값 이상)', transform=ax3.transAxes,
         fontsize=7, color='tomato', ha='center')
ax3.text(0.80, 0.15, '상대적 하락 예측\n(중앙값 미만)', transform=ax3.transAxes,
         fontsize=7, color='royalblue', ha='center')
ax3.text(0.5, -0.14, '주가 방향은 예측이 어려워 확률이 0.5 근처에 몰립니다 — 중앙값 기준으로 색 구분',
         transform=ax3.transAxes, ha='center', fontsize=7, color='gray')

# ── 특징맵 패널 (ax4) ──
ax4.text(0.5, 1.03,
         '필터마다 다른 날짜에서 강하게 반응해요',
         transform=ax4.transAxes, ha='center', fontsize=8, color='#333')
ax4.text(0.5, -0.14,
         '날짜 위치 (0=첫날, 29=마지막날)',
         transform=ax4.transAxes, ha='center', fontsize=7, color='gray')
ax4.set_ylabel('반응 강도')

plt.subplots_adjust(top=0.93)
plt.tight_layout()
ticker_tag = TICKER.replace(".", "_")
out_name = f"result/CnnTimeSeriesFeature_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ 1D CNN 시계열 특징 추출 실습 완료!\n")
