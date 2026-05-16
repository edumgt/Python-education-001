import numpy as np   # NaN(결측치) 값 표현을 위해 사용
import pandas as pd  # 테이블 형태의 데이터프레임 처리 라이브러리

# 포트폴리오 원본 데이터 (결측치 포함)
data = {
    '종목': ['A전자', 'B바이오', 'C에너지', 'D금융', 'E반도체', 'F플랫폼'],  # 종목명 리스트
    '섹터': ['IT', '헬스케어', '에너지', '금융', 'IT', '서비스'],             # 각 종목이 속한 섹터
    '보유수량': [15, 20, 35, 12, 18, 10],                                   # 보유 주식 수
    '매수가': [68000, 42000, np.nan, 51000, 73000, 98000],                  # 매수 단가 (C에너지는 데이터 누락)
    '현재가': [72000, 39800, 54000, np.nan, 75500, 101000],                 # 현재 주가 (D금융은 데이터 누락)
}

portfolio = pd.DataFrame(data)  # 딕셔너리를 데이터프레임으로 변환 (행=종목, 열=속성)
print("📌 원본 포트폴리오 데이터")
print(portfolio)  # 결측치(NaN) 포함 원본 데이터 출력

print("\n📌 결측치 개수")
print(portfolio.isnull().sum())  # 각 열별 NaN 개수 집계 출력 (매수가 1개, 현재가 1개)

# 결측치 보정(열 평균)
for col in ['매수가', '현재가']:                              # 매수가와 현재가 두 컬럼에 대해 반복
    portfolio[col] = portfolio[col].fillna(portfolio[col].mean())  # NaN을 해당 열의 평균값으로 대체

# 평가금액/수익률 계산
portfolio['평가금액'] = portfolio['보유수량'] * portfolio['현재가']  # 평가금액 = 보유수량 × 현재가
# 수익률(%) = (현재가 - 매수가) / 매수가 × 100
portfolio['수익률(%)'] = (portfolio['현재가'] - portfolio['매수가']) / portfolio['매수가'] * 100

print("\n📌 결측치 보정 + 파생지표")
print(portfolio)  # 결측치 보정 및 파생 컬럼이 추가된 데이터프레임 출력

# 섹터별로 그룹화하여 평가금액과 수익률의 평균을 계산
sector_summary = portfolio.groupby('섹터')[['평가금액', '수익률(%)']].mean()
print("\n📊 섹터별 평균 평가금액/수익률")
print(sector_summary)  # IT·금융·서비스 등 섹터별 집계 결과 출력
