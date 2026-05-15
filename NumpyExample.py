import numpy as np

# 가상 주가 시계열 (5개 종목 x 8일)
prices = np.array([
    [100, 101, 103, 102, 104, 106, 107, 109],
    [80, 79, 81, 82, 83, 84, 86, 87],
    [50, 51, 52, 54, 53, 55, 57, 58],
    [120, 118, 119, 121, 122, 124, 123, 125],
    [30, 31, 30, 32, 33, 34, 35, 36],
], dtype=float)

print("주가 행렬 shape:", prices.shape)
print("첫 번째 종목 주가:", prices[0])

# 일간 수익률 계산
returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
print("일간 수익률(첫 번째 종목):", returns[0])

# 종목별 평균 수익률/변동성
avg_returns = returns.mean(axis=1)
volatility = returns.std(axis=1)
print("종목별 평균 수익률:", avg_returns)
print("종목별 변동성:", volatility)

# 가중치 기반 포트폴리오 기대 수익률
weights = np.array([0.25, 0.2, 0.2, 0.25, 0.1])
portfolio_expected_return = np.dot(weights, avg_returns)
print("포트폴리오 기대 일간 수익률:", portfolio_expected_return)

# 마지막 날 기준 리밸런싱 금액 예시
capital = 1_000_000
last_prices = prices[:, -1]
allocation = capital * weights
shares = allocation / last_prices
print("종목별 매수 수량(가상):", shares)
