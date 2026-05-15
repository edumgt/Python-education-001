import numpy as np
import pandas as pd

# 포트폴리오 원본 데이터 (결측치 포함)
data = {
    '종목': ['A전자', 'B바이오', 'C에너지', 'D금융', 'E반도체', 'F플랫폼'],
    '섹터': ['IT', '헬스케어', '에너지', '금융', 'IT', '서비스'],
    '보유수량': [15, 20, 35, 12, 18, 10],
    '매수가': [68000, 42000, np.nan, 51000, 73000, 98000],
    '현재가': [72000, 39800, 54000, np.nan, 75500, 101000],
}

portfolio = pd.DataFrame(data)
print("📌 원본 포트폴리오 데이터")
print(portfolio)

print("\n📌 결측치 개수")
print(portfolio.isnull().sum())

# 결측치 보정(열 평균)
for col in ['매수가', '현재가']:
    portfolio[col] = portfolio[col].fillna(portfolio[col].mean())

# 평가금액/수익률 계산
portfolio['평가금액'] = portfolio['보유수량'] * portfolio['현재가']
portfolio['수익률(%)'] = (portfolio['현재가'] - portfolio['매수가']) / portfolio['매수가'] * 100

print("\n📌 결측치 보정 + 파생지표")
print(portfolio)

sector_summary = portfolio.groupby('섹터')[['평가금액', '수익률(%)']].mean()
print("\n📊 섹터별 평균 평가금액/수익률")
print(sector_summary)
