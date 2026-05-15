# Python 주식투자 머신러닝 실습 커리큘럼

이 저장소는 Python 기초부터 머신러닝/딥러닝까지 **주식투자 데이터 분석** 중심으로 학습하도록 구성되어 있습니다.

## 1) 기초 데이터 처리
- `NumpyExample.py`: 주가 배열, 수익률/변동성 계산, 포트폴리오 기대수익률
- `PandasExample.py`: 포트폴리오 결측치 처리, 평가금액/수익률 계산, 섹터별 집계

## 2) 회귀/특성 추출
- `LinearSample.py`: 거래량/변동성 기반 다음 날 수익률 회귀
- `HouseSample.py`: PER/PBR/ROE 기반 기대수익률 예측
- `FeatureSample.py`: 투자 뉴스 문장 TF-IDF 특징 추출

## 3) 분류 모델링
- `SvmSample.py`: 모멘텀/거래량 변화율 기반 매매 신호 분류
- `Digits.py`: RSI/MACD/변동성 기반 시장 국면(상승/하락) 분류

## 4) 고급 예제
- `BackpropStockSample.py`: 역전파를 직접 구현한 시계열 주가 예측
- `Word2Vec.py`: 투자 뉴스 토큰 임베딩 및 유사 단어 탐색
- `ImageMasking.py`: 수익률 히트맵에서 고위험 구간 마스킹

## 실행 방법
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install fonts-nanum
python -m pip install --upgrade pip
pip install pylint
```

## 결과 비교는 result 폴더의 파일과 합니다.
