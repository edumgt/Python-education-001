import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  SVM: 모멘텀/거래량 기반 매매 신호 분류 실습")
print("=" * 55)

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


def compute_returns(prices):
    return np.diff(prices.astype(float)) / prices[:-1]


def compute_volume_change(n):
    """가상 거래량 변화율 (실제 거래량 없으면 noise로 대체)"""
    return np.random.normal(0, 0.5, n)

# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/5] 매매 신호 데이터 준비 중...")
time.sleep(0.5)
np.random.seed(42)

if prices_raw is not None and len(prices_raw) > 50:
    returns = compute_returns(prices_raw)         # shape: (N-1,)
    vol_change = compute_volume_change(len(returns))  # shape: (N-1,)

    # 모멘텀: 오늘 수익률 (%)
    momentum = returns[:-1] * 100      # 오늘 모멘텀
    vol_feat = vol_change[:-1]         # 오늘 거래량 변화율
    # 레이블: 다음 날 수익률 > 0이면 매수(1)
    next_ret = returns[1:]
    y = (next_ret > 0).astype(int)

    # NaN 처리
    valid = ~(np.isnan(momentum) | np.isnan(vol_feat) | np.isnan(next_ret))
    momentum = momentum[valid]
    vol_feat = vol_feat[valid]
    y = y[valid]

    X = np.column_stack([momentum, vol_feat])
    buy_count = y.sum()
    print(f"   → 실제 데이터 {len(X)}개 샘플 (GS피앤엘 {TICKER})")
    print(f"      매수 신호(1): {buy_count}개  |  매도/보류(0): {len(y) - buy_count}개")
    data_source = f"GS피앤엘({TICKER}) 실제 데이터"
else:
    # ── fallback: 가상 데이터 ───────────────────────────────────────────────
    X = np.random.randn(240, 2)
    X[:, 0] = X[:, 0] * 2.0 + 0.5
    X[:, 1] = X[:, 1] * 1.5 - 0.2
    y = ((X[:, 0] > 0.3) & (X[:, 1] > -0.1)).astype(int)
    buy_count = y.sum()
    print(f"   → 총 {len(X)}개 샘플 생성 (가상 데이터)")
    print(f"      매수 신호(1): {buy_count}개  |  매도/보류(0): {len(y) - buy_count}개")
    data_source = "가상 데이터"

time.sleep(0.5)

print("\n[2/5] 학습/테스트 세트 분리 중 (7:3)...")
time.sleep(0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/5] SVM 선형 커널 모델 학습 중...")
print("   원리: 두 클래스 사이 경계선(초평면)을 찾아 마진을 최대화")
time.sleep(0.8)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print(f"   → 학습 완료!  서포트 벡터 수: {sum(model.n_support_)}개")
time.sleep(0.5)

print("\n[4/5] 테스트 세트 예측 & 정확도 평가 중...")
time.sleep(0.5)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
correct = (y_pred == y_test).sum()
print(f"   → {len(X_test)}개 중 {correct}개 정확히 분류")
print(f"   → 신호 분류 정확도: {acc:.4f} ({acc * 100:.1f}%)")
time.sleep(0.5)

print("\n[5/5] 결정 경계 시각화 중...")
time.sleep(0.5)


def plot_signal_boundary(features, labels, clf):
    step = max((features[:, 0].max() - features[:, 0].min()) / 200, 0.02)
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, grid_pred, cmap=plt.cm.coolwarm, alpha=0.25)
    ax.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm,
               edgecolors='k', alpha=0.6)
    ax.set_title(f"SVM 매매 신호 경계 (accuracy: {acc:.2f})\n({data_source})")
    ax.set_xlabel("모멘텀 (%)")
    ax.set_ylabel("거래량 변화율")

    # ── 초등학생도 이해할 수 있는 한글 설명 어노테이션 ──────────────────────────
    # x축 보충 설명
    ax.text(0.5, -0.13, '← 모멘텀이 낮을수록 힘이 없는 주식 / 높을수록 강하게 오르는 중 →',
            transform=ax.transAxes, ha='center', fontsize=7, color='gray')
    # y축 보충 설명
    ax.text(-0.18, 0.5, '위로 갈수록 거래량이 많이 늘어난 상태',
            transform=ax.transAxes, va='center', rotation=90, fontsize=7, color='gray')

    # 결정 경계선(중간 대각선)에 화살표 – 데이터 중심부 근방
    boundary_x = (x_min + x_max) / 2
    boundary_y = (y_min + y_max) / 2
    ax.annotate('이 선이 사야 할지/말아야 할지\n기준선이에요 (결정 경계)',
                xy=(boundary_x, boundary_y),
                xytext=(boundary_x - (x_max - x_min) * 0.4, boundary_y + (y_max - y_min) * 0.3),
                fontsize=7, color='#333',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 빨간(매수) 구역 설명
    ax.text(0.82, 0.88, '사야 할\n구역', transform=ax.transAxes,
            fontsize=8, color='darkred', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='tomato'))

    # 파란(매도) 구역 설명
    ax.text(0.13, 0.14, '팔거나\n기다릴\n구역', transform=ax.transAxes,
            fontsize=8, color='navy', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='royalblue'))

    # 빨간 점(매수 신호) 중 하나에 화살표
    buy_pts = features[labels == 1]
    if len(buy_pts) > 0:
        ax.annotate('매수 신호 점', xy=(buy_pts[0, 0], buy_pts[0, 1]),
                    xytext=(buy_pts[0, 0] + (x_max - x_min) * 0.15,
                            buy_pts[0, 1] - (y_max - y_min) * 0.2),
                    fontsize=7, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0))

    # 파란 점(매도 신호) 중 하나에 화살표
    sell_pts = features[labels == 0]
    if len(sell_pts) > 0:
        ax.annotate('매도 신호 점', xy=(sell_pts[0, 0], sell_pts[0, 1]),
                    xytext=(sell_pts[0, 0] - (x_max - x_min) * 0.25,
                            sell_pts[0, 1] + (y_max - y_min) * 0.2),
                    fontsize=7, color='navy',
                    arrowprops=dict(arrowstyle='->', color='navy', lw=1.0))

    # 전체 그래프 한 줄 요약
    fig.text(0.5, 0.995, '두 지표의 조합으로 \'선 하나\'를 그어 사야 할지 말아야 할지 나눕니다',
             ha='center', fontsize=9, color='#333', weight='bold', va='top')
    # ────────────────────────────────────────────────────────────────────────────

    plt.tight_layout()
    ticker_tag = TICKER.replace(".", "_")
    out_name = f"result/SvmTradeSignal_{ticker_tag}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"   → 그래프 저장: {out_name}")


plot_signal_boundary(X, y, model)
print("\n✓ SVM 매매 신호 분류 실습 완료!\n")
