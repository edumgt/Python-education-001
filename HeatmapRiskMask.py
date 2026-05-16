import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  히트맵 & 마스킹: 수익률 고위험 구간 탐지 실습")
print("=" * 55)

print("\n[1/5] 20일 × 20종목 가상 수익률 행렬 생성 중...")
time.sleep(0.5)
np.random.seed(42)
returns_map = np.random.normal(loc=0.1, scale=1.2, size=(20, 20))
print(f"   → 행렬 shape: {returns_map.shape}  (20일 × 20종목)")
print(f"   → 수익률 범위: {returns_map.min():.2f}% ~ {returns_map.max():.2f}%")
time.sleep(0.5)

print("\n[2/5] 고위험 구간 마스크 생성 중 (|수익률| > 2.0%)...")
print("   조건: 절대값이 2.0%를 넘으면 위험(1), 그 외 정상(0)")
time.sleep(0.5)
risk_mask = (np.abs(returns_map) > 2.0).astype(int)
risk_count = risk_mask.sum()
total = risk_mask.size
print(f"   → 위험 셀: {risk_count}개 / {total}개 ({risk_count / total * 100:.1f}%)")
time.sleep(0.3)

print("\n[3/5] 위험 구간 값 강조 처리 중...")
print("   방법: 위험 셀 값을 ±3.0으로 강제 설정 (시각적 대비 강화)")
time.sleep(0.5)
highlighted = returns_map.copy()
highlighted[risk_mask == 1] = np.sign(highlighted[risk_mask == 1]) * 3.0
print(f"   → 강조 전 최대 절대값: {np.abs(returns_map).max():.2f}%")
print(f"   → 강조 후 최대 절대값: {np.abs(highlighted).max():.2f}%")
time.sleep(0.3)

print("\n[4/5] 3패널 히트맵 시각화 구성 중...")
time.sleep(0.5)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(returns_map, cmap='RdYlGn')
axs[0].set_title("원본 수익률 맵")
print("   → 패널 1: 원본 수익률 히트맵")
time.sleep(0.3)

axs[1].imshow(risk_mask, cmap='gray')
axs[1].set_title("고위험 마스크")
print("   → 패널 2: 이진 위험 마스크 (흰색=위험)")
time.sleep(0.3)

axs[2].imshow(highlighted, cmap='RdYlGn')
axs[2].set_title("위험 구간 강조")
print("   → 패널 3: 강조 처리 히트맵")
time.sleep(0.3)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

print("\n[5/5] 그래프 저장 중...")
time.sleep(0.4)
plt.tight_layout()
plt.savefig("result/HeatmapRiskMask.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/HeatmapRiskMask.png")

print("\n✓ 히트맵 위험 마스킹 실습 완료!\n")
