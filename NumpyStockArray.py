import time

import numpy as np

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

print("\n✓ NumPy 실습 완료!\n")
