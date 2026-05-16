import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.makedirs("result", exist_ok=True)

print("=" * 60)
print("  PCA 차원 축소: 6차원 주식 지표 → 2차원 시각화 실습")
print("=" * 60)

print("\n[1/6] 50개 종목의 6가지 지표 생성 중...")
time.sleep(0.5)
np.random.seed(42)
n_stocks = 50
per = np.random.uniform(5, 25, n_stocks)
pbr = per * np.random.uniform(0.05, 0.12, n_stocks)
roe = np.random.uniform(5, 25, n_stocks)
volatility = np.random.uniform(0.1, 0.5, n_stocks)
avg_return = roe * 0.3 + np.random.normal(0, 1, n_stocks)
volume = np.random.uniform(10, 500, n_stocks)
X = np.column_stack([per, pbr, roe, volatility, avg_return, volume])
feature_names = ['PER', 'PBR', 'ROE', '변동성', '수익률', '거래량']
print(f"   → 데이터 형태: {X.shape}  ({n_stocks}종목 × {len(feature_names)}지표)")
print(f"   → 특성: {feature_names}")
time.sleep(0.5)

print("\n[2/6] StandardScaler로 표준화 중...")
print("   이유: 거래량(10~500)이 PBR(0.05~3) 보다 수백 배 커서 PCA 왜곡 방지")
time.sleep(0.5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   → 표준화 완료")
time.sleep(0.3)

print("\n[3/6] PCA 분석 실행 중 (6개 주성분 추출)...")
print("   원리: 데이터 분산(정보)이 가장 큰 방향을 순서대로 PC1, PC2... 추출")
time.sleep(0.8)
pca_full = PCA()
pca_full.fit(X_scaled)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
print("   → 분석 완료!")
time.sleep(0.3)

print("\n[4/6] 주성분별 설명 분산 비율 확인...")
time.sleep(0.4)
for i, (e, c) in enumerate(zip(explained, cumulative)):
    bar = "█" * int(e * 40)
    print(f"   PC{i + 1}: {e * 100:5.1f}%  누적={c * 100:.1f}%  {bar}")
    time.sleep(0.25)
print(f"\n   → PC1+PC2만으로도 정보의 {cumulative[1] * 100:.1f}% 보존!")
time.sleep(0.5)

print("\n[5/6] 6차원 → 2차원으로 압축 중...")
time.sleep(0.5)
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_scaled)
print(f"   원본: {X_scaled.shape} → 압축: {X_2d.shape}")
print("   (6개 지표를 2개 숫자로 요약 — 정보 손실 최소화)")
time.sleep(0.3)

print("\n[6/6] 시각화 중 (설명 분산 차트 + 2D 산점도)...")
time.sleep(0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(range(1, len(explained) + 1), explained, color='steelblue', alpha=0.7, label='개별 설명 분산')
ax1.step(range(1, len(cumulative) + 1), cumulative, where='mid', color='tomato', label='누적 설명 분산')
ax1.axhline(0.9, linestyle='--', color='gray', linewidth=0.8, label='90% 기준선')
ax1.set_xlabel("주성분(PC) 번호")
ax1.set_ylabel("설명 분산 비율")
ax1.set_title("PCA 설명 분산 비율")
ax1.legend()

sc = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=avg_return, cmap='RdYlGn',
                 alpha=0.8, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=ax2, label='수익률')
ax2.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
ax2.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
ax2.set_title("6차원 → 2차원 PCA 압축")

plt.tight_layout()
plt.savefig("result/PcaStockReduce.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/PcaStockReduce.png")

print("\n✓ PCA 차원 축소 실습 완료!\n")
