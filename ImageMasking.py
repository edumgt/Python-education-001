import matplotlib.pyplot as plt  # 이미지 및 그래프 시각화 라이브러리
import numpy as np               # 배열 연산 및 마스킹 처리 라이브러리
import korean_font  # noqa: F401 # 한글 폰트 설정 모듈 (임포트 시 폰트 자동 적용)

# 1) 20일 x 20종목 수익률 히트맵(가상)
np.random.seed(42)  # 난수 시드 고정으로 재현 가능한 데이터 생성
# 평균 0.1%(약보합), 표준편차 1.2%인 정규분포로 20×20 수익률 행렬 생성
returns_map = np.random.normal(loc=0.1, scale=1.2, size=(20, 20))

# 2) 변동성 위험 구간 마스크 (절대수익률 > 2.0)
# 절대값이 2.0%를 초과하는 셀을 위험 구간(1)으로, 그 외는 정상(0)으로 분류한 이진 마스크
risk_mask = (np.abs(returns_map) > 2.0).astype(int)

# 3) 위험 구간 강조
highlighted = returns_map.copy()  # 원본을 복사하여 원본 보존
# 위험 구간(마스크=1)의 값을 부호는 유지한 채 절대값 3.0으로 강제 설정 → 시각적으로 도드라지게 강조
highlighted[risk_mask == 1] = np.sign(highlighted[risk_mask == 1]) * 3.0

# 4) 시각화
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1행 3열 서브플롯 생성 (가로 12인치, 세로 4인치)

axs[0].imshow(returns_map, cmap='RdYlGn')  # 첫 번째 패널: 원본 수익률 히트맵 (빨강=손실, 초록=수익)
axs[0].set_title("원본 수익률 맵")         # 첫 번째 패널 제목

axs[1].imshow(risk_mask, cmap='gray')     # 두 번째 패널: 위험 마스크 (흰색=위험, 검정=정상)
axs[1].set_title("고위험 마스크")          # 두 번째 패널 제목

axs[2].imshow(highlighted, cmap='RdYlGn')  # 세 번째 패널: 위험 구간이 강조된 수익률 맵
axs[2].set_title("위험 구간 강조")          # 세 번째 패널 제목

for ax in axs:       # 세 패널 모두에 대해 반복
    ax.set_xticks([])  # x축 눈금 제거 (종목 번호 숫자가 불필요하므로 생략)
    ax.set_yticks([])  # y축 눈금 제거 (일자 번호 숫자가 불필요하므로 생략)

plt.tight_layout()  # 서브플롯 간 여백 자동 조정으로 겹침 방지
plt.savefig("ImageMasking.png", dpi=150, bbox_inches="tight")  # 고해상도 PNG로 저장
print("그래프 저장: ImageMasking.png")  # 저장 완료 메시지 출력
