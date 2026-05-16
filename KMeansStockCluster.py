import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("result", exist_ok=True)

print("=" * 60)
print("  K-Means 군집화: 주식 유형 자동 분류 실습 (비지도 학습)")
print("=" * 60)

print("\n[1/6] 가상 주식 데이터 생성 중 (60개 종목)...")
print("   ※ 정답 라벨 없음 — 비지도 학습!")
time.sleep(0.5)
np.random.seed(42)
returns_groups = [
    np.random.normal(0.15, 0.03, 20),   # 성장주: 고수익
    np.random.normal(0.07, 0.015, 20),  # 가치주: 중간 수익
    np.random.normal(0.04, 0.008, 20),  # 방어주: 저수익
]
vol_groups = [
    np.random.normal(0.30, 0.05, 20),   # 성장주: 고변동
    np.random.normal(0.15, 0.03, 20),   # 가치주: 중간 변동
    np.random.normal(0.08, 0.02, 20),   # 방어주: 저변동
]
avg_returns = np.concatenate(returns_groups)
volatility = np.concatenate(vol_groups)
X = np.column_stack([avg_returns, volatility])
print(f"   → {len(X)}개 종목  |  특성: [평균 수익률, 변동성]")
time.sleep(0.5)

print("\n[2/6] StandardScaler로 데이터 표준화 중...")
print("   이유: 수익률과 변동성의 스케일을 통일해야 거리 계산이 공정해짐")
time.sleep(0.5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   → 표준화 완료 (평균=0, 표준편차=1)")
time.sleep(0.3)

print("\n[3/6] K-Means 군집화 실행 중 (k=3)...")
print("   Step 1: 랜덤 위치에 k개의 중심점(centroid) 배치")
print("   Step 2: 각 점을 가장 가까운 중심점의 군집에 할당")
print("   Step 3: 각 군집의 평균으로 중심점 이동")
print("   Step 4: 중심점이 더 이상 움직이지 않을 때까지 2~3 반복")
time.sleep(1.0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
print(f"   → 군집화 완료!  수렴까지 반복 횟수: {kmeans.n_iter_}회")
time.sleep(0.3)

print("\n[4/6] 군집 중심점 해석 중 (수익률 기준 이름 부여)...")
time.sleep(0.5)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
order = np.argsort(centers_original[:, 0])[::-1]
label_names_map = {order[0]: '성장주', order[1]: '가치주', order[2]: '방어주'}
label_names = [label_names_map[l] for l in labels]
for name, center in zip(['성장주', '가치주', '방어주'], centers_original[order]):
    count = label_names.count(name)
    print(f"   {name}: 중심 수익률={center[0]:.3f}  중심 변동성={center[1]:.3f}  → {count}개 종목")
    time.sleep(0.3)

print("\n[5/6] 군집별 통계 요약...")
time.sleep(0.4)
for name in ['성장주', '가치주', '방어주']:
    mask = np.array([n == name for n in label_names])
    print(f"   {name}: {mask.sum()}종목  |  "
          f"평균수익률={avg_returns[mask].mean():.3f}  "
          f"평균변동성={volatility[mask].mean():.3f}")
    time.sleep(0.2)

print("\n[6/6] 군집 시각화 중...")
time.sleep(0.5)
colors = {'성장주': 'tomato', '가치주': 'royalblue', '방어주': 'seagreen'}
plt.figure(figsize=(7, 5))
for name, color in colors.items():
    mask = [n == name for n in label_names]
    plt.scatter(X[mask, 0], X[mask, 1], label=name, color=color,
                alpha=0.7, edgecolors='k', linewidths=0.4)
centers_inv = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers_inv[:, 0], centers_inv[:, 1], marker='X', s=200,
            color='black', zorder=5, label='군집 중심')
plt.xlabel("평균 수익률")
plt.ylabel("변동성")
plt.title("K-Means 군집화: 주식 유형 자동 분류")
plt.legend()
plt.tight_layout()
plt.savefig("result/KMeansStockCluster.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/KMeansStockCluster.png")

print("\n✓ K-Means 군집화 실습 완료!\n")
