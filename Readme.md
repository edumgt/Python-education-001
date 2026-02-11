# Python and ML Education Curriculum

## 📘 1단계. Python 기반 데이터/머신러닝 기초

### 1-1. Python 프로그래밍 기초
- 변수, 조건문, 반복문, 함수
- 리스트, 딕셔너리, 클래스, 모듈 사용법
- numpy, pandas, matplotlib 기본

### 1-2. 데이터 분석 & 시각화
- 데이터 불러오기 (CSV, Excel, 이미지 등)
- EDA(탐색적 데이터 분석) 실습
- seaborn, plotly, pandas-profiling 활용

### 1-3. 머신러닝 기본 이론
- 지도학습 vs 비지도학습 개념
- 분류, 회귀, 군집화 개요
- Train/Test Split, K-Fold, 일반화/과적합 이해

### 1-4. Scikit-Learn 실습
- 분류: 로지스틱 회귀, KNN, 결정트리
- 회귀: 선형회귀, 랜덤포레스트 회귀
- 성능 평가 (Confusion Matrix, ROC AUC 등)
- 하이퍼파라미터 튜닝 (GridSearchCV)

---

## 📗 2단계. 딥러닝 및 이미지 처리

### 2-1. 딥러닝 개념
- 인공신경망(ANN) 기본 구조
- 활성화 함수, 손실 함수, 옵티마이저
- 과적합 방지: Dropout, EarlyStopping

### 2-2. TensorFlow/Keras 실습
- Sequential 모델로 손글씨(MNIST) 분류
- CNN(합성곱 신경망)으로 이미지 분류
- 모델 저장, 로딩, 배치 학습

### 2-3. 이미지 처리 + OpenCV
- 이미지 읽기, 자르기, 필터링
- 얼굴 인식, 윤곽선 추출
- OpenCV + NumPy 조합 실습
- OpenCV + GPU(CUDA) 간단 연동

---

## 📕 3단계. 생성형 AI 기반 이미지 생성

### 3-1. 생성형 AI 이해
- GAN, VAE, Diffusion 개념 비교
- 텍스트 → 이미지: Stable Diffusion 원리
- HuggingFace diffusers 소개

### 3-2. Stable Diffusion 실습
- diffusers 설치 및 파이프라인 실행
- 텍스트→이미지 (text2img)
- 이미지→이미지 (img2img)
- 제어형 생성 (ControlNet, pose2img)

### 3-3. 실전 프로젝트
- 나만의 프롬프트 라이브러리 만들기
- 실시간 프롬프트 → 이미지 저장 시스템
- 생성 이미지 → OpenCV로 편집 or 필터링
- 결과물 ZIP / PDF / 슬라이드 자동 저장

---

## 💼 보너스: 배포 및 연계
- Streamlit or Gradio로 웹앱 만들기
- 이미지 생성 자동화 파이프라인 구성
- GPU 환경 설정: Colab / 로컬 CUDA / Docker

---

## ✅ 단계별 구성 방식 (총 3단계)

| 단계 | 이름               | 키워드                              |
|------|--------------------|--------------------------------------|
| 1단계 | Python + ML        | numpy, pandas, sklearn               |
| 2단계 | 딥러닝 + CV        | keras, tensorflow, opencv, cnn       |
| 3단계 | 생성형 AI 이미지   | diffusers, torch, ControlNet         |

---

## 📦 패키지 요약표

| 분류       | 주요 패키지                                        |
|------------|-----------------------------------------------------|
| ML 기본    | pandas, numpy, scikit-learn, matplotlib             |
| 딥러닝     | tensorflow, keras, torch (선택)                    |
| 이미지처리 | opencv-python, PIL                                  |
| 이미지 생성| diffusers, transformers, accelerate, safetensors    |

---

## 💻 개발 환경 초기 설정 (Windows)

```bash
# 가상환경 생성
C:\Python312\python.exe -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# pip 업그레이드
python.exe -m pip install --upgrade pip

# ML 패키지 설치
pip install gensim
pip install scikit-learn

# requirements.txt 가 있을 경우
pip install -r requirements.txt
