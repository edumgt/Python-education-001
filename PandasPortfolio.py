import time

import numpy as np
import pandas as pd

print("=" * 55)
print("  Pandas 포트폴리오 데이터 처리 실습")
print("=" * 55)

print("\n[1/5] 포트폴리오 원본 데이터 로딩 (결측치 포함)...")
time.sleep(0.5)
data = {
    '종목':     ['A전자', 'B바이오', 'C에너지', 'D금융', 'E반도체', 'F플랫폼'],
    '섹터':     ['IT', '헬스케어', '에너지', '금융', 'IT', '서비스'],
    '보유수량': [15, 20, 35, 12, 18, 10],
    '매수가':   [68000, 42000, np.nan, 51000, 73000, 98000],
    '현재가':   [72000, 39800, 54000, np.nan, 75500, 101000],
}
portfolio = pd.DataFrame(data)
print(portfolio.to_string(index=False))
time.sleep(0.5)

print("\n[2/5] 결측치 탐지 중...")
time.sleep(0.5)
null_count = portfolio.isnull().sum()
found = null_count[null_count > 0].to_dict()
print(f"   → 결측치 발견: {found}")
time.sleep(0.3)

print("\n[3/5] 결측치 → 열 평균으로 보정 중...")
time.sleep(0.5)
for col in ['매수가', '현재가']:
    mean_val = portfolio[col].mean()
    missing = portfolio[col].isnull().sum()
    portfolio[col] = portfolio[col].fillna(mean_val)
    print(f"   {col}: {missing}개 NaN → 평균 {mean_val:,.0f}원으로 대체")
    time.sleep(0.3)

print("\n[4/5] 평가금액 & 수익률(%) 계산 중...")
print("   평가금액 = 보유수량 × 현재가")
print("   수익률(%) = (현재가 - 매수가) / 매수가 × 100")
time.sleep(0.5)
portfolio['평가금액'] = portfolio['보유수량'] * portfolio['현재가']
portfolio['수익률(%)'] = (portfolio['현재가'] - portfolio['매수가']) / portfolio['매수가'] * 100
print(portfolio[['종목', '평가금액', '수익률(%)']].to_string(index=False))
time.sleep(0.5)

print("\n[5/5] 섹터별 그룹화 & 집계 중 (groupby)...")
time.sleep(0.5)
sector_summary = portfolio.groupby('섹터')[['평가금액', '수익률(%)']].mean()
print(sector_summary.round(2))

print("\n✓ Pandas 실습 완료!\n")
