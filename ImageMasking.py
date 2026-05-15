import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform

# 운영체제별 한글 폰트 설정
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')  # 리눅스용

matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 1. 기본 이미지 생성 (256x256, 흑백 그레이디언트)
image = np.tile(np.linspace(0, 255, 256), (256, 1))

# 2. 마스크 생성 (원형 부분만 강조)
mask = np.zeros_like(image)
cx, cy, r = 128, 128, 60  # 중심과 반지름
y_indices, x_indices = np.ogrid[:256, :256]
distance = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
mask[distance <= r ** 2] = 1

# 3. 이미지 마스킹: 원 내부만 밝게 처리
highlighted = image.copy()
highlighted[mask == 1] = 255  # 밝은 값으로 변경

# 4. 시각화
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("원본 이미지")
axs[1].imshow(mask, cmap='gray')
axs[1].set_title("마스크")
axs[2].imshow(highlighted, cmap='hot')
axs[2].set_title("마스킹 및 색상 강조")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
