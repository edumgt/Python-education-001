import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.makedirs("../result", exist_ok=True)

print("=" * 55)
print("  SVM: RSI/MACD 기반 시장 국면(상승/하락) 분류 실습")
print("=" * 55)

# ── 1. 실제 주가 데이터 로드 ───────────────────────────────
print("\n[1/5] 주가 데이터 로드 & 기술지표 계산 중 (RSI, MACD, 변동성)...")
time.sleep(0.5)
np.random.seed(42)
TICKER = '078935.KS'
prices_raw = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices_raw = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices_raw)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices_raw is None:
    days = 500
    t = np.arange(days, dtype=float)
    prices_raw = (100 + 0.1 * t + 10 * np.sin(t / 40) + np.random.normal(0, 2, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")


def compute_rsi(prices, n=14):
    s = pd.Series(prices.astype(float))
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, 1e-8)
    return (100 - 100 / (1 + rs)).values


def compute_macd(prices, fast=12, slow=26):
    s = pd.Series(prices.astype(float))
    return (s.ewm(span=fast).mean() - s.ewm(span=slow).mean()).values


def compute_volatility(prices, n=10):
    s = pd.Series(prices.astype(float))
    return s.pct_change().rolling(n).std().values


rsi_vals  = compute_rsi(prices_raw)
macd_vals = compute_macd(prices_raw)
vol_vals  = compute_volatility(prices_raw)

# 다음 날 수익률 기반 레이블 (1=상승, 0=하락)
returns = np.diff(prices_raw) / prices_raw[:-1]
labels  = (returns > 0).astype(int)

# NaN 제거 (첫 ~26일은 MACD 계산 불가)
valid_idx = ~(np.isnan(rsi_vals[:-1]) | np.isnan(macd_vals[:-1]) | np.isnan(vol_vals[:-1]))
rsi_v   = rsi_vals[:-1][valid_idx]
macd_v  = macd_vals[:-1][valid_idx]
vol_v   = vol_vals[:-1][valid_idx]
y       = labels[valid_idx]

X = np.column_stack([rsi_v, macd_v, vol_v])
print(f"   → {len(X)}개 샘플  |  상승장(1): {y.sum()}개  |  하락장(0): {(y==0).sum()}개")
time.sleep(0.5)

print("\n[2/5] 학습/테스트 세트 분리 중 (8:2)...")
time.sleep(0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/5] SVM RBF 커널 모델 학습 중...")
print("   RBF = 방사형 기저 함수 (곡선 경계로 복잡한 패턴 학습)")
print("   gamma=0.2: 결정 경계의 곡률 조절 (작을수록 완만)")
time.sleep(0.8)
clf = SVC(gamma=0.2)
clf.fit(X_train, y_train)
print(f"   → 학습 완료!  서포트 벡터 수: {sum(clf.n_support_)}개")
time.sleep(0.5)

print("\n[4/5] 테스트 세트 예측 & 성능 평가 중...")
time.sleep(0.5)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"   → 정확도: {acc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['하락장', '상승장']))
time.sleep(0.5)

print("\n[5/5] 예측 결과 시각화 중 (RSI vs MACD)...")
time.sleep(0.5)
points = X_test[:40]
preds = y_pred[:40]
colors = np.where(preds == 1, 'tomato', 'royalblue')
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)
ax.set_title(f"RSI-MACD 기반 시장 국면 예측 ({TICKER})")
ax.set_xlabel("RSI")
ax.set_ylabel("MACD")

# ── 초등학생도 이해할 수 있는 한글 설명 어노테이션 ──────────────────────────
# x축 보충 설명
ax.text(0.5, -0.18, 'RSI → 70 이상이면 \'너무 많이 올랐다\' 신호 / 30 이하면 \'너무 많이 내렸다\' 신호',
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')
# y축 보충 설명
ax.text(-0.2, 0.5, 'MACD → 양수면 상승 추세, 음수면 하락 추세',
        transform=ax.transAxes, va='center', rotation=90, fontsize=7, color='gray')

# 빨간 점 밀집 구역(상승장) 설명
bull_pts = points[preds == 1]
if len(bull_pts) > 0:
    bx, by = bull_pts[:, 0].mean(), bull_pts[:, 1].mean()
    ax.text(bx, by + 0.25, '상승장 구역 📈', fontsize=8, color='darkred', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='tomato'))
    # 빨간 점 하나에 화살표
    ax.annotate('상승장\n예측 점', xy=(bull_pts[0, 0], bull_pts[0, 1]),
                xytext=(bull_pts[0, 0] + 5, bull_pts[0, 1] - 0.6),
                fontsize=7, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0))

# 파란 점 밀집 구역(하락장) 설명
bear_pts = points[preds == 0]
if len(bear_pts) > 0:
    dx, dy = bear_pts[:, 0].mean(), bear_pts[:, 1].mean()
    ax.text(dx, dy - 0.25, '하락장 구역 📉', fontsize=8, color='navy', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='royalblue'))
    # 파란 점 하나에 화살표
    ax.annotate('하락장\n예측 점', xy=(bear_pts[0, 0], bear_pts[0, 1]),
                xytext=(bear_pts[0, 0] - 8, bear_pts[0, 1] + 0.6),
                fontsize=7, color='navy',
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.0))

# 전체 그래프 한 줄 요약
fig.text(0.5, 0.995, 'RSI와 MACD 두 지표로 지금 시장이 오르는 중인지 내리는 중인지 판단합니다',
         ha='center', fontsize=9, color='#333', weight='bold', va='top')
# ────────────────────────────────────────────────────────────────────────────

plt.tight_layout()
ticker_tag = TICKER.replace('.', '_')
out_name = f"../result/SvmMarketPhase_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ SVM 시장 국면 분류 실습 완료!\n")
