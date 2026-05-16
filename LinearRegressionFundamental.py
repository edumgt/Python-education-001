import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  선형 회귀: PER/PBR/ROE → 종목 기대 수익률 예측")
print("=" * 55)

print("\n[1/5] 종목 펀더멘털 데이터 로딩 중...")
time.sleep(0.5)
data = pd.DataFrame({
    'PER': [8.1, 10.5, 7.8, 15.2, 11.3, 9.6, 12.4, 6.9, 14.7, 8.8],
    'PBR': [0.8, 1.2, 0.7, 1.8, 1.1, 0.9, 1.4, 0.6, 1.7, 0.85],
    'ROE': [14.2, 12.5, 16.1, 9.2, 11.8, 13.7, 10.1, 17.4, 8.6, 15.3],
    '기대수익률': [9.5, 7.2, 10.4, 4.8, 6.9, 8.1, 5.9, 11.0, 4.5, 9.0],
})
print(data.to_string(index=False))
time.sleep(0.5)

print("\n[2/5] 학습/테스트 세트 분리 중 (8:2)...")
time.sleep(0.5)
X = data[['PER', 'PBR', 'ROE']]
y = data['기대수익률']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   → 학습: {len(X_train)}개, 테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/5] 선형 회귀 모델 학습 중...")
print("   수식: 기대수익률 = w0 + w1×PER + w2×PBR + w3×ROE")
time.sleep(0.8)
model = LinearRegression()
model.fit(X_train, y_train)
print("   → 학습 완료!")
for feat, coef in zip(['PER', 'PBR', 'ROE'], model.coef_):
    print(f"      {feat} 계수: {coef:.4f}")
    time.sleep(0.2)

print("\n[4/5] 테스트 세트 성능 평가 중 (MAE)...")
time.sleep(0.5)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
for real, pred in zip(y_test, y_pred):
    print(f"   실제={real:.1f}%  예측={pred:.2f}%  오차={abs(real - pred):.2f}%")
    time.sleep(0.2)
print(f"   → 평균 절대 오차(MAE): {mae:.4f}%")
time.sleep(0.3)

print("\n[5/5] 새 종목 기대 수익률 예측 중 (PER=9.2, PBR=0.95, ROE=14.8)...")
time.sleep(0.5)
new_stock = pd.DataFrame({'PER': [9.2], 'PBR': [0.95], 'ROE': [14.8]})
predicted = model.predict(new_stock)
print(f"   → 예상 기대 수익률: {predicted[0]:.2f}%")

print("\n[6/5] 결과 시각화 중 (실제 vs 예측 수익률 산점도)...")
time.sleep(0.3)

# 전체 데이터로 예측값 계산 (시각화용)
y_all_pred = model.predict(X)
y_all = y.values

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_all, y_all_pred, c='steelblue', edgecolors='k', linewidths=0.5,
           zorder=3, label='회사 데이터 점')

# 완벽한 예측일 때의 대각선 (y = x)
lim_min = min(y_all.min(), y_all_pred.min()) - 0.5
lim_max = max(y_all.max(), y_all_pred.max()) + 0.5
diag = np.linspace(lim_min, lim_max, 100)
ax.plot(diag, diag, color='tomato', linewidth=2, linestyle='--', label='완벽한 예측선')

ax.set_xlabel("실제 기대 수익률 (%)")
ax.set_ylabel("예측 기대 수익률 (%)")
ax.set_title("선형 회귀: PER/PBR/ROE → 기대 수익률 예측")
ax.legend()

# ── 초등학생도 이해할 수 있는 한글 설명 어노테이션 ──────────────────────────
# 각 점 설명 – 첫 번째 점에 화살표
ax.annotate('점 하나 = 회사 한 곳\n(실제와 예측 수익률 비교)',
            xy=(y_all[0], y_all_pred[0]),
            xytext=(y_all[0] + 0.8, y_all_pred[0] - 0.8),
            fontsize=7, color='#333',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 회귀(대각)선에 화살표
mid = len(diag) // 2
ax.annotate('이 직선으로 미래 수익률을 예측해요\n(점이 선에 가까울수록 예측이 정확!)',
            xy=(diag[mid], diag[mid]),
            xytext=(diag[mid] + 1.2, diag[mid] - 1.0),
            fontsize=7, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='tomato'))

# x축 보충 설명
ax.text(0.5, -0.13, '실제로 올랐던 수익률 (정답) →  오른쪽일수록 더 많이 오른 회사',
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')
# y축 보충 설명
ax.text(-0.22, 0.5, '컴퓨터가 예측한 수익률 →  위로 갈수록 더 오를 거라고 예측',
        transform=ax.transAxes, va='center', rotation=90, fontsize=7, color='gray')

# 전체 그래프 한 줄 요약
fig.text(0.5, 0.995,
         'PER·PBR 같은 회사 성적표로 주가 상승률을 예측합니다',
         ha='center', fontsize=9, color='#333', weight='bold', va='top')
# ────────────────────────────────────────────────────────────────────────────

plt.tight_layout()
plt.savefig("result/LinearRegressionFundamental.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/LinearRegressionFundamental.png")

print("\n✓ 펀더멘털 선형 회귀 실습 완료!\n")
