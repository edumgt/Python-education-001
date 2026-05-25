import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 65)
print("  오토인코더: 주가 이상 탐지 실습")
print("=" * 65)
print()
print("  오토인코더 핵심 아이디어:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  인코더: 입력(30일 수익률) → 압축 표현(잠재 벡터)   │")
print("  │  디코더: 잠재 벡터         → 복원(30일 수익률)       │")
print("  │                                                       │")
print("  │  정상 패턴 → 복원 오차 작음 (잘 배웠으니 잘 복원)   │")
print("  │  이상 패턴 → 복원 오차 큼  (본 적 없는 패턴)        │")
print("  │                                                       │")
print("  │  활용: 급락/급등/거래량 폭발 등 이상 시점 자동 탐지  │")
print("  └──────────────────────────────────────────────────────┘")

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")

# ── 1. 주가 데이터 로드 ───────────────────────────────────────
print("\n[1/9] 주가 데이터 로드 중 (078935.KS - GS피앤엘)...")
time.sleep(0.5)

TICKER = '078935.KS'
prices = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2018-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 100:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    days = 700
    t    = np.arange(days, dtype=float)
    base = 10000 + 30 * t + 600 * np.sin(t / 50) + np.random.normal(0, 100, days)
    # 인위적 이상 구간 삽입 (급락 2회, 급등 1회)
    base[200:210] -= 1500
    base[450:455] += 2000
    base[600:608] -= 1800
    prices = base.astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성 (이상 구간 3곳 포함)")

returns = np.diff(prices) / (prices[:-1] + 1e-8)
print(f"   → {len(prices)}일치 주가  |  수익률: {returns.min():.3f} ~ {returns.max():.3f}")
time.sleep(0.3)

# ── 2. 슬라이딩 윈도우 데이터셋 ──────────────────────────────
print("\n[2/9] 슬라이딩 윈도우 데이터셋 구성 중...")
print("   윈도우 크기: 30일  |  각 윈도우 z-score 정규화")
time.sleep(0.5)

SEQ_LEN = 30

windows, dates_idx = [], []
for i in range(len(returns) - SEQ_LEN):
    w    = returns[i:i + SEQ_LEN]
    mu, sd = w.mean(), w.std() + 1e-8
    windows.append((w - mu) / sd)
    dates_idx.append(i + SEQ_LEN)   # 윈도우 끝 날짜 인덱스

X_all = np.array(windows, dtype=np.float32)   # (N, 30)
print(f"   → 전체 윈도우 수: {len(X_all)}")
time.sleep(0.3)

# ── 3. 학습/테스트 분리 ───────────────────────────────────────
print("\n[3/9] 학습(정상 구간) / 전체 재구성 분리 중...")
print("   정상 구간: 일별 수익률 절대값이 3σ 미만인 윈도우만 학습에 사용")
print("   → 오토인코더는 정상 패턴만 보고 '정상이란 무엇인지' 학습합니다")
time.sleep(0.5)

sigma_thr = 3 * returns.std()
is_normal = np.array([
    np.all(np.abs(returns[i:i + SEQ_LEN]) < sigma_thr)
    for i in range(len(returns) - SEQ_LEN)
])
X_train = torch.tensor(X_all[is_normal]).to(DEVICE)
X_full  = torch.tensor(X_all).to(DEVICE)
print(f"   → 정상 학습 샘플: {len(X_train)}  |  전체(재구성용): {len(X_full)}")
time.sleep(0.3)

# ── 4. 오토인코더 모델 정의 ───────────────────────────────────
print("\n[4/9] 오토인코더 모델 정의 중...")
print("   인코더: 30 → 16 → 8 (잠재 공간)")
print("   디코더:  8 → 16 → 30 (복원)")
time.sleep(0.5)


class StockAutoencoder(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, seq_len),
        )

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


model = StockAutoencoder().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"   → 총 파라미터: {total_params:,}개")
print(model)
time.sleep(0.4)

# ── 5. 학습 설정 ─────────────────────────────────────────────
print("\n[5/9] 학습 설정 중 (MSE Loss + Adam)...")
time.sleep(0.4)

criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
EPOCHS     = 200
BATCH_SIZE = 32
print(f"   → 에폭={EPOCHS}  배치={BATCH_SIZE}  손실=재구성 MSE")
time.sleep(0.3)

# ── 6. 학습 루프 ─────────────────────────────────────────────
print("\n[6/9] 오토인코더 학습 시작 (정상 데이터만)...")
print("   정상 패턴을 압축하고 복원하는 방법을 학습합니다")
time.sleep(0.8)

loss_history = []
prev_loss    = None
model.train()
for epoch in range(EPOCHS):
    perm       = torch.randperm(len(X_train))
    epoch_loss = 0.0
    for start in range(0, len(X_train), BATCH_SIZE):
        idx  = perm[start:start + BATCH_SIZE]
        xb   = X_train[idx]
        optimizer.zero_grad()
        x_hat = model(xb)
        loss  = criterion(x_hat, xb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    avg = epoch_loss / len(X_train)
    loss_history.append(avg)
    if epoch % 50 == 0:
        trend = ""
        if prev_loss is not None:
            trend = " ↓" if avg < prev_loss else " →"
        print(f"   Epoch {epoch:4d} | 재구성 손실: {avg:.6f}{trend}")
        prev_loss = avg
        time.sleep(0.2)

print(f"   Epoch {EPOCHS:4d} | 학습 완료!")
time.sleep(0.4)

# ── 7. 재구성 오차 계산 & 임계값 설정 ────────────────────────
print("\n[7/9] 전체 구간 재구성 오차 계산 & 이상 탐지 임계값 설정 중...")
time.sleep(0.4)

model.eval()
with torch.no_grad():
    x_hat_all = model(X_full).cpu().numpy()
    x_all_np  = X_full.cpu().numpy()

recon_errors = np.mean((x_all_np - x_hat_all) ** 2, axis=1)   # (N,)

# 학습 데이터의 재구성 오차 분포로 임계값 결정 (95 percentile)
train_errors = recon_errors[is_normal]
threshold    = np.percentile(train_errors, 95)
anomaly_mask = recon_errors > threshold
n_anomaly    = anomaly_mask.sum()

print(f"   → 정상 재구성 오차 평균:  {train_errors.mean():.6f}")
print(f"   → 이상 탐지 임계값(95%): {threshold:.6f}")
print(f"   → 이상 탐지된 윈도우 수: {n_anomaly}  ({n_anomaly/len(recon_errors)*100:.1f}%)")
time.sleep(0.3)

# ── 8. 잠재 공간 분석 ────────────────────────────────────────
print("\n[8/9] 잠재 공간 표현 분석 중...")
time.sleep(0.3)

with torch.no_grad():
    latent_all = model.encode(X_full).cpu().numpy()   # (N, 8)

# 잠재 벡터의 첫 두 차원으로 산점도 (PCA 없이 직접)
print(f"   → 잠재 벡터 차원: {latent_all.shape[1]}  |  정상:{(~anomaly_mask).sum()}  이상:{anomaly_mask.sum()}")
time.sleep(0.3)

# ── 9. 시각화 ─────────────────────────────────────────────────
print("\n[9/9] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(16, 12))

fig.text(0.5, 0.98,
         "오토인코더 이상 탐지 — 정상 패턴 학습 후 '낯선 움직임'을 자동으로 찾아냅니다",
         ha='center', fontsize=10, color='#333', weight='bold')

# 학습 손실 곡선
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(loss_history, color='steelblue', linewidth=1.5)
ax1.set_title("재구성 손실 학습 곡선 (MSE)")
ax1.set_xlabel("에폭")
ax1.set_ylabel("재구성 MSE")
ax1.grid(alpha=0.3)
ax1.annotate('처음엔 복원을 못해요', xy=(0, loss_history[0]),
             xytext=(20, loss_history[0] * 0.85),
             fontsize=7, color='darkred',
             arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('점차 정상 패턴을\n잘 복원해요', xy=(EPOCHS-1, loss_history[-1]),
             xytext=(EPOCHS * 0.55, loss_history[-1] + (loss_history[0]-loss_history[-1]) * 0.3),
             fontsize=7, color='navy',
             arrowprops=dict(arrowstyle='->', color='gray'))
ax1.text(0.5, -0.18, '손실이 낮아질수록 정상 패턴을 잘 기억한 것',
         transform=ax1.transAxes, ha='center', fontsize=7, color='gray')

# 전체 재구성 오차 시계열
ax2 = fig.add_subplot(3, 1, 2)
x_axis = np.array(dates_idx)
ax2.plot(x_axis, recon_errors, color='steelblue', linewidth=0.8, alpha=0.6, label='재구성 오차')
ax2.axhline(threshold, linestyle='--', color='red', linewidth=1.2, label=f'임계값={threshold:.5f}')
ax2.fill_between(x_axis, threshold, recon_errors,
                 where=anomaly_mask, color='tomato', alpha=0.5, label='이상 탐지 구간')
ax2.set_title(f"전체 구간 재구성 오차  |  이상 {n_anomaly}개 탐지 (임계={threshold:.5f})")
ax2.set_xlabel("날짜 인덱스")
ax2.set_ylabel("재구성 MSE")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.text(0.5, -0.10,
         '빨간 구간 = 모델이 본 적 없는 패턴 (급락·급등 등 이상 시점)',
         transform=ax2.transAxes, ha='center', fontsize=7.5, color='gray')

# 정상 샘플 복원 예시
ax3 = fig.add_subplot(3, 3, 4)
normal_idxs = np.where(~anomaly_mask)[0]
if len(normal_idxs) > 0:
    ni = normal_idxs[len(normal_idxs) // 2]
    ax3.plot(x_all_np[ni], color='steelblue', linewidth=1.5, label='실제')
    ax3.plot(x_hat_all[ni], color='tomato', linewidth=1.5, linestyle='--', label='복원')
    ax3.set_title(f"정상 구간 복원 예시 (오차={recon_errors[ni]:.5f})")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
    ax3.text(0.5, -0.18, '두 선이 거의 겹쳐요 — 정상 패턴은 잘 복원됩니다',
             transform=ax3.transAxes, ha='center', fontsize=7, color='gray')

# 이상 샘플 복원 예시
ax4 = fig.add_subplot(3, 3, 5)
anom_idxs = np.where(anomaly_mask)[0]
if len(anom_idxs) > 0:
    ai  = anom_idxs[np.argmax(recon_errors[anom_idxs])]  # 최대 오차 이상 샘플
    ax4.plot(x_all_np[ai], color='steelblue', linewidth=1.5, label='실제')
    ax4.plot(x_hat_all[ai], color='tomato', linewidth=1.5, linestyle='--', label='복원')
    ax4.set_title(f"이상 구간 복원 예시 (오차={recon_errors[ai]:.5f})")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)
    ax4.text(0.5, -0.18, '두 선이 많이 달라요 — 본 적 없는 패턴이라 복원 실패',
             transform=ax4.transAxes, ha='center', fontsize=7, color='gray')

# 잠재 공간 산점도 (첫 두 차원)
ax5 = fig.add_subplot(3, 3, 6)
ax5.scatter(latent_all[~anomaly_mask, 0], latent_all[~anomaly_mask, 1],
            c='steelblue', s=8, alpha=0.4, label='정상')
ax5.scatter(latent_all[anomaly_mask, 0], latent_all[anomaly_mask, 1],
            c='tomato', s=20, alpha=0.8, label='이상', marker='x', linewidths=1.5)
ax5.set_title("잠재 공간 산점도 (첫 2차원)")
ax5.set_xlabel("잠재 차원 1")
ax5.set_ylabel("잠재 차원 2")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)
ax5.text(0.5, -0.18, '이상 데이터(×)는 정상 군집(●)과 떨어져 있어요',
         transform=ax5.transAxes, ha='center', fontsize=7, color='gray')

# 재구성 오차 히스토그램
ax6 = fig.add_subplot(3, 3, 7)
ax6.hist(train_errors, bins=30, color='steelblue', alpha=0.7, label='정상')
ax6.hist(recon_errors[anomaly_mask], bins=15, color='tomato', alpha=0.7, label='이상')
ax6.axvline(threshold, linestyle='--', color='black', linewidth=1.2, label=f'임계값')
ax6.set_title("재구성 오차 분포")
ax6.set_xlabel("재구성 MSE")
ax6.set_ylabel("빈도")
ax6.legend(fontsize=7)
ax6.grid(alpha=0.3)
ax6.text(0.5, -0.18, '두 분포가 잘 분리될수록 탐지 성능이 좋습니다',
         transform=ax6.transAxes, ha='center', fontsize=7, color='gray')

# 모델 구조 설명
ax7 = fig.add_subplot(3, 3, 8)
ax7.axis('off')
struct = (
    "오토인코더 구조\n\n"
    f"입력: 30일 수익률\n"
    f"(z-score 정규화)\n\n"
    "[ 인코더 ]\n"
    "Linear(30 → 16)\n"
    "ReLU\n"
    "Linear(16 → 8)\n"
    "   ↓ 잠재 벡터(8차원)\n\n"
    "[ 디코더 ]\n"
    "Linear(8 → 16)\n"
    "ReLU\n"
    "Linear(16 → 30)\n"
    "   ↓ 복원 시퀀스(30일)\n\n"
    "재구성 오차 = MSE(입력, 복원)\n"
    f"이상 임계값: 95 퍼센타일\n"
    f"파라미터: {total_params:,}개"
)
ax7.text(0.05, 0.95, struct, transform=ax7.transAxes,
         fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 실제 주가 위에 이상 시점 표시
ax8 = fig.add_subplot(3, 3, 9)
plot_prices = prices[1:]   # returns와 길이 맞추기
ax8.plot(range(len(plot_prices)), plot_prices, color='steelblue',
         linewidth=0.8, alpha=0.7, label='주가')
# 이상 탐지 윈도우 끝 날짜에 빨간 세로선
for idx in dates_idx:
    if idx < len(anomaly_mask) and anomaly_mask[dates_idx.index(idx)]:
        ax8.axvspan(idx - SEQ_LEN, idx, color='tomato', alpha=0.15)
ax8.set_title("주가 위 이상 탐지 구간 표시")
ax8.set_xlabel("날짜 인덱스")
ax8.set_ylabel("주가")
ax8.grid(alpha=0.3)
ax8.text(0.5, -0.18, '붉은 구간 = 오토인코더가 이상으로 판단한 시기',
         transform=ax8.transAxes, ha='center', fontsize=7, color='gray')

plt.subplots_adjust(top=0.94, hspace=0.50, wspace=0.35)
ticker_tag = TICKER.replace('.', '_')
out_name   = f"../result/AutoencoderAnomalyDetect_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ 오토인코더 주가 이상 탐지 실습 완료!\n")
