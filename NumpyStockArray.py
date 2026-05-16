import numpy as np  # 고성능 수치 배열 연산 라이브러리

# 가상 주가 시계열 (5개 종목 x 8일)
prices = np.array([
    [100, 101, 103, 102, 104, 106, 107, 109],  # 종목 A: 8일간 주가
    [80,  79,  81,  82,  83,  84,  86,  87 ],  # 종목 B: 8일간 주가
    [50,  51,  52,  54,  53,  55,  57,  58 ],  # 종목 C: 8일간 주가
    [120, 118, 119, 121, 122, 124, 123, 125],  # 종목 D: 8일간 주가
    [30,  31,  30,  32,  33,  34,  35,  36 ],  # 종목 E: 8일간 주가
], dtype=float)  # 수익률 계산 시 소수점 처리를 위해 float 타입 지정

print("주가 행렬 shape:", prices.shape)       # 행렬 크기 출력 → (5, 8): 5종목 × 8일
print("첫 번째 종목 주가:", prices[0])        # 종목 A의 8일치 주가 출력

# 일간 수익률 계산
# prices[:, 1:]: 2일~8일 / prices[:, :-1]: 1일~7일 → (오늘 - 어제) / 어제 = 일간 수익률
returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
print("일간 수익률(첫 번째 종목):", returns[0])  # 종목 A의 7일간 일간 수익률 출력

# 종목별 평균 수익률/변동성
avg_returns = returns.mean(axis=1)  # axis=1: 열(일자) 방향으로 평균 → 종목별 평균 일간 수익률
volatility = returns.std(axis=1)    # axis=1: 열(일자) 방향으로 표준편차 → 종목별 수익률 변동성
print("종목별 평균 수익률:", avg_returns)  # 각 종목의 평균 일간 수익률 출력
print("종목별 변동성:", volatility)        # 각 종목의 수익률 표준편차(변동성) 출력

# 가중치 기반 포트폴리오 기대 수익률
weights = np.array([0.25, 0.2, 0.2, 0.25, 0.1])  # 5개 종목의 투자 비중 (합계=1.0)
# 내적(dot product): 각 종목의 가중치 × 평균 수익률을 합산 → 포트폴리오 전체 기대 수익률
portfolio_expected_return = np.dot(weights, avg_returns)
print("포트폴리오 기대 일간 수익률:", portfolio_expected_return)  # 가중 평균 기대 수익률 출력

# 마지막 날 기준 리밸런싱 금액 예시
capital = 1_000_000           # 총 투자 원금: 100만 원 (밑줄 구분자는 가독성용)
last_prices = prices[:, -1]   # 각 종목의 마지막 날(8일차) 주가 추출
allocation = capital * weights # 각 종목에 배분할 금액 = 원금 × 비중
shares = allocation / last_prices  # 배분 금액을 주가로 나눠 매수 가능 수량 계산
print("종목별 매수 수량(가상):", shares)  # 리밸런싱 시 각 종목의 매수 주수 출력
