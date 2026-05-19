import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 55)
print("  NumPy 주가 배열 & 포트폴리오 계산 실습")
print("=" * 55)

print("\n[1/5] 가상 주가 데이터 생성 중...")
time.sleep(0.5)
prices = np.array([
    [100, 101, 103, 102, 104, 106, 107, 109],
    [80,  79,  81,  82,  83,  84,  86,  87 ],
    [50,  51,  52,  54,  53,  55,  57,  58 ],
    [120, 118, 119, 121, 122, 124, 123, 125],
    [30,  31,  30,  32,  33,  34,  35,  36 ],
], dtype=float)
print(f"   → 주가 행렬 shape: {prices.shape}  (5종목 × 8일)")
time.sleep(0.3)

print("\n[2/5] 일간 수익률 계산 중...")
print("   수식: (오늘 주가 - 어제 주가) / 어제 주가")
time.sleep(0.5)
returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
print(f"   → 종목 A 7일간 수익률: {returns[0].round(4)}")
time.sleep(0.3)

print("\n[3/5] 종목별 평균 수익률 & 변동성 계산 중...")
time.sleep(0.5)
avg_returns = returns.mean(axis=1)
volatility = returns.std(axis=1)
for i, (r, v) in enumerate(zip(avg_returns, volatility)):
    print(f"   종목 {'ABCDE'[i]}: 평균 수익률={r:.4f}  변동성={v:.4f}")
    time.sleep(0.2)

print("\n[4/5] 포트폴리오 기대 수익률 계산 중 (가중치 내적)...")
print("   수식: Σ(가중치_i × 평균수익률_i)")
time.sleep(0.5)
weights = np.array([0.25, 0.2, 0.2, 0.25, 0.1])
portfolio_expected_return = np.dot(weights, avg_returns)
print(f"   → 가중치: {weights}")
print(f"   → 포트폴리오 기대 일간 수익률: {portfolio_expected_return:.5f}")
time.sleep(0.3)

print("\n[5/5] 100만원 기준 종목별 리밸런싱 수량 계산 중...")
time.sleep(0.5)
capital = 1_000_000
last_prices = prices[:, -1]
allocation = capital * weights
shares = allocation / last_prices
for i, (s, p) in enumerate(zip(shares, last_prices)):
    print(f"   종목 {'ABCDE'[i]}: 현재가={p:6.0f}원  →  {s:.2f}주 매수")
    time.sleep(0.2)

print("\n[6/5] 결과 시각화 중...")
time.sleep(0.3)

labels = list('ABCDE')
days = [f"Day{i+1}" for i in range(8)]
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('NumPy 주가 배열 & 포트폴리오 분석', fontsize=12, weight='bold')

# 패널 1: 주가 추이
ax = axes[0]
for i, label in enumerate(labels):
    ax.plot(days, prices[i], marker='o', markersize=4, color=colors[i], label=label)
ax.set_title('5종목 8일 주가 추이')
ax.set_xlabel('거래일')
ax.set_ylabel('주가 (원)')
ax.legend(title='종목', fontsize=8)
ax.tick_params(axis='x', labelsize=8)

# 패널 2: 평균 수익률 & 변동성 비교
ax = axes[1]
x = np.arange(len(labels))
bar_w = 0.35
bars1 = ax.bar(x - bar_w/2, avg_returns * 100, bar_w, color='steelblue', label='평균 수익률(%)')
bars2 = ax.bar(x + bar_w/2, volatility * 100, bar_w, color='tomato', alpha=0.8, label='변동성(%)')
ax.set_title('종목별 평균 수익률 & 변동성')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('%')
ax.legend(fontsize=8)
ax.axhline(0, color='black', linewidth=0.7)

# 패널 3: 포트폴리오 가중치 파이차트
ax = axes[2]
wedges, texts, autotexts = ax.pie(
    weights, labels=labels, autopct='%1.0f%%',
    colors=colors, startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(8)
ax.set_title(f'포트폴리오 가중치\n(기대 일간 수익률: {portfolio_expected_return*100:.4f}%)')

plt.tight_layout()
plt.savefig("../result/NumpyStockArray.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/NumpyStockArray.png")

print("\n✓ NumPy 실습 완료!\n")
