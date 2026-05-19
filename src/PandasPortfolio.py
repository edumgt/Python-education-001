import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

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

print("\n[6/5] 결과 시각화 중...")
time.sleep(0.3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Pandas 포트폴리오 데이터 분석', fontsize=12, weight='bold')

# 패널 1: 종목별 수익률 (양수=초록, 음수=빨간)
ax = axes[0]
returns = portfolio['수익률(%)'].values
bar_colors = ['seagreen' if r >= 0 else 'tomato' for r in returns]
bars = ax.barh(portfolio['종목'], returns, color=bar_colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('수익률 (%)')
ax.set_title('종목별 수익률\n(초록: 수익 / 빨간: 손실)')
for bar, val in zip(bars, returns):
    ax.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=8)

# 패널 2: 섹터별 평균 평가금액 & 수익률
ax2 = axes[1]
sectors = sector_summary.index.tolist()
x = np.arange(len(sectors))
bar_w = 0.35
ax2.bar(x - bar_w/2, sector_summary['평가금액'] / 10000, bar_w,
        color='steelblue', label='평균 평가금액 (만원)')
ax2.set_ylabel('평균 평가금액 (만원)', color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')

ax3 = ax2.twinx()
ax3.bar(x + bar_w/2, sector_summary['수익률(%)'], bar_w,
        color='darkorange', alpha=0.85, label='평균 수익률(%)')
ax3.set_ylabel('평균 수익률 (%)', color='darkorange')
ax3.tick_params(axis='y', labelcolor='darkorange')
ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')

ax2.set_xticks(x)
ax2.set_xticklabels(sectors, fontsize=9)
ax2.set_title('섹터별 평균 평가금액 & 수익률')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig("../result/PandasPortfolio.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/PandasPortfolio.png")

print("\n✓ Pandas 실습 완료!\n")
