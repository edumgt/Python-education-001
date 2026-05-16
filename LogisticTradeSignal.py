import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("result", exist_ok=True)

print("=" * 60)
print("  로지스틱 회귀: RSI/MACD 기반 매수/매도 신호 분류 실습")
print("=" * 60)

# ── 실제 주가 데이터 다운로드 ────────────────────────────────────────────────
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


def compute_returns(prices):
    return np.diff(prices.astype(float)) / prices[:-1]

# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/6] RSI & MACD 데이터 준비 중...")
time.sleep(0.5)
np.random.seed(42)

if prices_raw is not None and len(prices_raw) > 50:
    rsi_full = compute_rsi(prices_raw)      # shape: (N,)
    macd_full = compute_macd(prices_raw)    # shape: (N,)
    returns_full = compute_returns(prices_raw)  # shape: (N-1,)

    # 레이블: 다음 날 수익률 > 0이면 1 (매수), 그 외 0
    # rsi/macd는 인덱스 0~N-1, 다음 날 수익률은 인덱스 0~N-2 (즉 prices[i+1]/prices[i]-1)
    # 정렬: rsi[i], macd[i] → label = (prices[i+1] > prices[i])
    rsi_feat = rsi_full[:-1]     # 오늘 RSI
    macd_feat = macd_full[:-1]   # 오늘 MACD
    y_all = (returns_full > 0).astype(int)  # 다음 날 수익률 > 0이면 매수

    # NaN 처리
    valid = ~(np.isnan(rsi_feat) | np.isnan(macd_feat) | np.isnan(y_all.astype(float)))
    rsi = rsi_feat[valid]
    macd = macd_feat[valid]
    y = y_all[valid]

    n = len(rsi)
    print(f"   → 실제 데이터 {n}개 샘플  |  매수(1): {y.sum()}개  |  매도/관망(0): {(y == 0).sum()}개")
    data_source = f"GS피앤엘({TICKER}) 실제 데이터"
else:
    # ── fallback: 가상 데이터 ───────────────────────────────────────────────
    n = 300
    rsi = np.random.uniform(20, 80, n)
    macd = np.random.normal(0, 1, n)
    y = ((rsi > 55) & (macd > 0.1)).astype(int)
    noise_idx = np.random.choice(n, size=int(n * 0.08), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    print(f"   → {n}개 샘플  |  매수(1): {y.sum()}개  |  매도/관망(0): {(y == 0).sum()}개")
    print(f"   (노이즈 {int(n * 0.08)}개 추가 → 현실적인 경계 모사)")
    data_source = "가상 데이터"

time.sleep(0.5)

print("\n[2/6] 학습/테스트 세트 분리 중 (75:25)...")
time.sleep(0.4)
X = np.column_stack([rsi, macd])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/6] StandardScaler로 표준화 중...")
print("   이유: RSI(20~80)와 MACD(-3~3)의 스케일 차이 제거")
time.sleep(0.5)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("   → 표준화 완료 (평균=0, 표준편차=1)")
time.sleep(0.3)

print("\n[4/6] 로지스틱 회귀 모델 학습 중...")
print("   수식: P(매수) = sigmoid(w0 + w1×RSI + w2×MACD)")
print("   sigmoid: 어떤 숫자든 0~1 확률로 변환")
time.sleep(0.8)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
print("   → 학습 완료!")
print(f"      RSI 계수: {model.coef_[0][0]:.4f}  |  MACD 계수: {model.coef_[0][1]:.4f}")
print(f"      절편: {model.intercept_[0]:.4f}")
time.sleep(0.5)

print("\n[5/6] 테스트 세트 평가 중...")
time.sleep(0.5)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(classification_report(y_test, y_pred, target_names=['매도/관망', '매수']))
print(f"   ROC-AUC: {auc:.4f}  (1.0에 가까울수록 완벽한 분류)")
time.sleep(0.5)

print("\n[6/6] 매수 확률 결정 경계 시각화 중...")
time.sleep(0.5)
step = 0.5
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5))
cf = ax.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.4)
plt.colorbar(cf, ax=ax, label='매수 확률')
buy_mask = y == 1
ax.scatter(X[buy_mask, 0], X[buy_mask, 1], c='tomato', label='매수(1)',
           edgecolors='k', linewidths=0.4, alpha=0.7)
ax.scatter(X[~buy_mask, 0], X[~buy_mask, 1], c='royalblue', label='매도/관망(0)',
           edgecolors='k', linewidths=0.4, alpha=0.7)
ax.set_xlabel("RSI")
ax.set_ylabel("MACD")
ax.set_title(f"로지스틱 회귀: 매수/매도 신호 분류\n({data_source})")
ax.legend()

# ── 초등학생도 이해할 수 있는 한글 설명 어노테이션 ──────────────────────────
# x축 보충 설명
ax.text(0.5, -0.13, 'RSI → 숫자가 클수록 주가가 많이 오른 상태 (70 이상이면 과열!)',
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')
# y축 보충 설명
ax.text(-0.18, 0.5, 'MACD → 양수(+)면 오르는 중, 음수(-)면 내리는 중',
        transform=ax.transAxes, va='center', rotation=90, fontsize=7, color='gray')

# 초록(매수) 구역 설명
ax.text(0.78, 0.88, '여기선\n매수 확률이\n높아요', transform=ax.transAxes,
        fontsize=8, color='darkgreen', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))

# 빨간(매도/관망) 구역 설명
ax.text(0.15, 0.18, '여기선\n매도/관망이\n나아요', transform=ax.transAxes,
        fontsize=8, color='navy', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='royalblue'))

# 매수 신호 점 하나에 화살표
buy_idx_list = np.where(buy_mask)[0]
if len(buy_idx_list) > 5:
    buy_idx = buy_idx_list[5]
    ax.annotate('매수 신호 점', xy=(X[buy_idx, 0], X[buy_idx, 1]),
                xytext=(X[buy_idx, 0] - 8, X[buy_idx, 1] + 0.8),
                fontsize=7, color='#333',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.0))

# 전체 그래프 한 줄 요약
fig.text(0.5, 0.995, 'RSI·MACD 지표로 \'살까? 말까?\' 확률을 계산합니다',
         ha='center', fontsize=9, color='#333', weight='bold', va='top')
# ────────────────────────────────────────────────────────────────────────────

plt.tight_layout()
ticker_tag = TICKER.replace(".", "_")
out_name = f"result/LogisticTradeSignal_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ 로지스틱 회귀 실습 완료!\n")
