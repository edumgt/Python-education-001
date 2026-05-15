import matplotlib.pyplot as plt
import numpy as np

# 1) 20일 x 20종목 수익률 히트맵(가상)
np.random.seed(42)
returns_map = np.random.normal(loc=0.1, scale=1.2, size=(20, 20))

# 2) 변동성 위험 구간 마스크 (절대수익률 > 2.0)
risk_mask = (np.abs(returns_map) > 2.0).astype(int)

# 3) 위험 구간 강조
highlighted = returns_map.copy()
highlighted[risk_mask == 1] = np.sign(highlighted[risk_mask == 1]) * 3.0

# 4) 시각화
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(returns_map, cmap='RdYlGn')
axs[0].set_title("원본 수익률 맵")

axs[1].imshow(risk_mask, cmap='gray')
axs[1].set_title("고위험 마스크")

axs[2].imshow(highlighted, cmap='RdYlGn')
axs[2].set_title("위험 구간 강조")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
