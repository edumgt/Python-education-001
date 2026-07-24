# 로컬 Python ML/DL → AWS · GCP 대응 가이드

> 한 줄 요약: 이 저장소의 23개 실습 파일(+ webapp)에서 쓰인 Python 라이브러리·기법을 기준으로, 같은 작업을 AWS와 GCP의 관리형 서비스로 옮길 때 무엇을 쓰고 어떻게 사용하는지 정리한 문서예요.

이 문서는 다른 `docs/*.md` 파일들과 달리 **특정 소스 파일 하나를 설명하는 문서가 아니라, 커리큘럼 전체를 클라우드 관점에서 가로지르는 참고 문서**입니다. `Readme.md`의 1~7단계 학습 로드맵과 동일한 순서로 구성했습니다.

---

## 범례

| 구분 | 의미 |
|------|------|
| **Local Python** | 이 저장소의 `src/*.py` 코드가 실제로 하는 일 |
| **AWS** | 동일 작업에 대응하는 AWS 관리형 서비스와 사용법 |
| **GCP** | 동일 작업에 대응하는 GCP 관리형 서비스와 사용법 |

---

## 전체 매핑표

| 단계 | 파일 | 핵심 기법 | AWS 대응 리소스 | GCP 대응 리소스 |
|------|------|-----------|-----------------|-----------------|
| 1 | `NumpyStockArray.py` | 배열 연산, 수익률/변동성 | SageMaker Studio Notebook | Vertex AI Workbench |
| 1 | `PandasPortfolio.py` | DataFrame, groupby, 결측치 | SageMaker Processing (Pandas) | BigQuery / Dataproc Serverless |
| 1 | `YfinanceNormalize.py` | 시세 수집, 정규화 | Lambda + EventBridge → S3 | Cloud Functions + Scheduler → BigQuery |
| 2 | `LinearRegressionFundamental.py` | 선형 회귀 | SageMaker Linear Learner | BigQuery ML `LINEAR_REG` |
| 2 | `LinearRegressionReturn.py` | 회귀 기반 시계열 예측 | SageMaker Linear Learner | Vertex AI AutoML Tables (회귀) |
| 2 | `LogisticTradeSignal.py` | 로지스틱 회귀 분류 | SageMaker Linear Learner (분류모드) | BigQuery ML `LOGISTIC_REG` |
| 3 | `SvmTradeSignal.py` / `SvmMarketPhase.py` | SVM (linear / RBF) | SageMaker Script Mode (SKLearn Estimator) | Vertex AI Custom Training (sklearn 컨테이너) |
| 3 | `KMeansStockCluster.py` | K-Means 군집 | SageMaker 내장 K-Means | BigQuery ML `KMEANS` |
| 3 | `PcaStockReduce.py` | PCA 차원축소 | SageMaker 내장 PCA | BigQuery ML `PCA` |
| 3 | `HyperparamTuning.py` | GridSearch, K-Fold | SageMaker Automatic Model Tuning | Vertex AI Vizier / Hyperparameter Tuning |
| 4 | `TimeSeriesAnalysis.py` | ACF/PACF, ADF 검정 | SageMaker Notebook + statsmodels | Vertex AI Workbench + statsmodels |
| 4 | `ArimaStockForecast.py` | ARIMA(2,1,2) | Amazon Forecast / SageMaker DeepAR | BigQuery ML `ARIMA_PLUS` |
| 4 | `HeatmapRiskMask.py` | 히트맵/마스킹 시각화 | QuickSight | Looker Studio |
| 5 | `NeuralNetBackprop.py` | NumPy 역전파 직접구현 | SageMaker Studio Notebook (교육용) | Vertex AI Workbench (교육용) |
| 5 | `RnnBackprop.py` | 바닐라 RNN + BPTT | SageMaker Studio Notebook (교육용) | Vertex AI Workbench (교육용) |
| 6 | `LstmStockPyTorch.py` | LSTM 시계열 예측 | SageMaker PyTorch Estimator (GPU) | Vertex AI Training (PyTorch 사전빌드 컨테이너) |
| 6 | `TimeSeriesWindow.py` | CLI LSTM | SageMaker Training Job (하이퍼파라미터 인자화) | Vertex AI Custom Job (args 전달) |
| 6 | `CnnTimeSeriesFeature.py` | 1D CNN | SageMaker PyTorch Estimator | Vertex AI Training |
| 6 | `CnnCandleChart.py` | 2D CNN 이미지 분류 | SageMaker PyTorch + S3 이미지 데이터셋 | Vertex AI Training + Cloud Storage 데이터셋 |
| 6 | `CnnLstmHybrid.py` | CNN+LSTM 하이브리드 | SageMaker PyTorch (커스텀 스크립트) | Vertex AI Training (커스텀 컨테이너) |
| 7 | `TransformerAttention.py` | Multi-Head Self-Attention | SageMaker PyTorch + Distributed Data Parallel | Vertex AI Training + TPU/Multi-GPU |
| + | `AutoencoderAnomalyDetect.py` | Autoencoder 이상탐지 | SageMaker Random Cut Forest / PyTorch AE | BigQuery ML `AUTOENCODER` |
| + | `DqnTradingAgent.py` | DQN 강화학습 트레이딩 | Ray RLlib on SageMaker Training | Ray on Vertex AI |
| + | `webapp/backend` (Flask) | 모델 서빙 API | SageMaker Endpoint / App Runner | Cloud Run |

---

## 1단계 — 데이터 처리: NumPy · Pandas · yfinance

`NumpyStockArray.py`, `PandasPortfolio.py`, `YfinanceNormalize.py`에 해당합니다. 로컬에서는 pandas로 시세를 받아 배열 연산·정규화를 하지만, 클라우드에서는 "수집(스케줄) → 저장 → 분산 전처리"로 역할이 나뉩니다.

**Local Python**
```python
import yfinance as yf
df = yf.download("005930.KS")
df["norm"] = (df.Close - df.Close.min()) / (df.Close.max() - df.Close.min())
```

**AWS — Lambda + EventBridge + S3 + SageMaker Processing**
- EventBridge 규칙(cron)이 매일 Lambda를 트리거해 시세를 받아 S3에 적재
- 대용량 정규화·피처엔지니어링은 SageMaker Processing Job에서 pandas 스크립트 그대로 실행
- SQL 조회가 필요하면 Glue Crawler → Athena
```bash
aws events put-rule --name daily-pull \
  --schedule-expression "cron(0 22 * * ? *)"
```
```python
processor = SKLearnProcessor(
    framework_version="1.2-1", role=role,
    instance_type="ml.m5.xlarge", instance_count=1)
processor.run(code="normalize.py",
    inputs=[ProcessingInput(source="s3://bkt/raw")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/out")])
```

**GCP — Cloud Functions + Scheduler + BigQuery**
- Cloud Scheduler → Cloud Functions(Python)가 시세 수집 후 BigQuery에 적재
- 정규화는 BigQuery SQL(윈도우 함수) 또는 Dataproc Serverless(Spark pandas API)
- Vertex AI Workbench 노트북에서 `pandas-gbq`로 그대로 로컬 코드 재사용 가능
```bash
gcloud scheduler jobs create http daily-pull \
  --schedule="0 22 * * *" --uri=$FUNCTION_URL
```
```sql
SELECT ticker, close,
  (close - MIN(close) OVER w) / (MAX(close) OVER w - MIN(close) OVER w) AS norm
FROM `proj.stocks.prices`
WINDOW w AS (PARTITION BY ticker)
```

> **핵심 차이:** AWS는 이벤트(EventBridge)+컴퓨트(Lambda/Processing)+오브젝트스토리지(S3)+쿼리(Athena)를 개별 서비스로 조합하고, GCP는 BigQuery가 저장소와 SQL 엔진을 겸해 정규화 로직 상당수를 SQL로 옮길 수 있습니다.

---

## 2단계 — 회귀·분류: Linear/Logistic Regression · SVM

`LinearRegressionFundamental.py`, `LinearRegressionReturn.py`, `LogisticTradeSignal.py`, `SvmTradeSignal.py`, `SvmMarketPhase.py`에 해당합니다.

**Local Python**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
pred = model.predict(X_test)
```

**AWS — SageMaker Linear Learner / Script Mode**
- 선형·로지스틱 회귀는 **Linear Learner** 내장 알고리즘으로 대체 (`predictor_type`: regressor/binary_classifier)
- SVM처럼 내장 알고리즘이 없는 경우 **SKLearn Estimator(Script Mode)**로 기존 코드를 거의 그대로 실행
```python
from sagemaker.sklearn.estimator import SKLearn
est = SKLearn(entry_point="svm_train.py", role=role,
    framework_version="1.2-1", instance_type="ml.m5.large")
est.fit({"train": "s3://bkt/train.csv"})
```

**GCP — BigQuery ML / Vertex AI Custom Training**
- 회귀·분류는 SQL 한 줄로 학습 가능한 **BigQuery ML**이 가장 빠른 경로
- SVM 등 sklearn 그대로 쓰려면 Vertex AI Custom Training + 사전빌드 sklearn 컨테이너
```sql
CREATE OR REPLACE MODEL `proj.ds.trade_signal`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['signal'])
AS SELECT rsi, macd, signal FROM `proj.ds.features`;
```
```bash
gcloud ai custom-jobs create --region=us-central1 \
  --worker-pool-spec=replica-count=1,machine-type=n1-standard-4,\
executor-image-uri=us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest,\
local-package-path=./svm,script=svm_train.py
```

---

## 3단계 — 비지도학습: K-Means · PCA

`KMeansStockCluster.py`, `PcaStockReduce.py`에 해당합니다.

**Local Python**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
km = KMeans(n_clusters=3).fit(X)
Z = PCA(n_components=2).fit_transform(X)
```

**AWS — SageMaker 내장 K-Means / PCA**

둘 다 SageMaker 내장 알고리즘(Protobuf recordIO 입력)으로 대규모 데이터에서도 분산 학습됩니다.
```python
from sagemaker import KMeans
kmeans = KMeans(role=role, instance_count=1,
    instance_type="ml.m5.large", k=3)
kmeans.fit(kmeans.record_set(train_data))
```

**GCP — BigQuery ML `KMEANS` / `PCA`**
```sql
CREATE OR REPLACE MODEL `proj.ds.cluster`
OPTIONS(model_type='KMEANS', num_clusters=3)
AS SELECT * EXCEPT(ticker) FROM `proj.ds.features`;

CREATE OR REPLACE MODEL `proj.ds.pca2d`
OPTIONS(model_type='PCA', num_principal_components=2)
AS SELECT * FROM `proj.ds.features`;
```

---

## 3단계 — 하이퍼파라미터 튜닝: GridSearchCV · K-Fold

`HyperparamTuning.py`에 해당합니다.

**Local Python**
```python
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), {"C": [0.01, 0.1, 1, 10]}, cv=5)
gs.fit(X, y)
```

**AWS — SageMaker Automatic Model Tuning (HPO)**

격자탐색 대신 베이지안 최적화로 더 적은 시도로 최적값을 찾고, 여러 학습 job을 병렬 실행합니다.
```python
tuner = HyperparameterTuner(
    estimator=est, objective_metric_name="validation:accuracy",
    hyperparameter_ranges={"C": ContinuousParameter(0.001, 1000)},
    max_jobs=20, max_parallel_jobs=4)
tuner.fit({"train": s3_train})
```

**GCP — Vertex AI Hyperparameter Tuning (Vizier)**
```bash
gcloud ai hp-tuning-jobs create \
  --region=us-central1 --display-name=svm-tune \
  --config=hptuning_config.yaml \
  --max-trial-count=20 --parallel-trial-count=4
```
내부적으로 Google의 블랙박스 최적화 엔진 **Vertex AI Vizier**가 탐색 전략을 결정합니다.

---

## 4단계 — 시계열: statsmodels ARIMA

`TimeSeriesAnalysis.py`, `ArimaStockForecast.py`에 해당합니다.

**Local Python**
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(2, 1, 2)).fit()
forecast = model.forecast(steps=20)
```

**AWS — Amazon Forecast / SageMaker DeepAR**
- **Amazon Forecast**: 완전관리형, ARIMA/ETS/Prophet/DeepAR+를 자동 비교해 최적 알고리즘 선택
- 커스텀 딥러닝 시계열이면 **SageMaker DeepAR** 내장 알고리즘
```bash
aws forecast create-dataset-import-job ...
aws forecast create-predictor --predictor-name arima_stock \
  --forecast-horizon 20 --input-data-config ... \
  --algorithm-arn arn:aws:forecast:::algorithm/ARIMA
```

**GCP — BigQuery ML `ARIMA_PLUS`**

SQL 한 줄로 계절성·휴일 효과까지 자동 처리하는 ARIMA_PLUS 모델을 학습·예측합니다.
```sql
CREATE OR REPLACE MODEL `proj.ds.arima_close`
OPTIONS(model_type='ARIMA_PLUS', time_series_timestamp_col='date',
  time_series_data_col='close') AS
SELECT date, close FROM `proj.ds.prices`;

SELECT * FROM ML.FORECAST(MODEL `proj.ds.arima_close`,
  STRUCT(20 AS horizon));
```

> 더 정교한 자동화가 필요하면 **Vertex AI Forecasting(AutoML)**도 대안이며, ARIMA_PLUS보다 다변량·외생변수 처리에 유리합니다.

---

## 5단계 — 신경망 직접구현: NumPy Backprop · RNN

`NeuralNetBackprop.py`, `RnnBackprop.py`는 프레임워크 없이 순전파/역전파/BPTT를 직접 구현하는 교육용 코드라 **1:1 대응 관리형 서비스가 없습니다.** 클라우드의 역할은 "실행 환경 제공"으로 좁혀집니다.

- **Local Python**: 순수 NumPy. 체인룰, 기울기 소실을 직접 눈으로 확인하기 위한 교육 목적이며 프레임워크·클라우드 종속성이 없음
- **AWS**: SageMaker Studio Notebook (CPU) — GPU가 필요 없는 소규모 연산이므로 `ml.t3.medium` 노트북 인스턴스로 충분. 학습 목적이면 로컬 대비 이점이 크지 않음
- **GCP**: Vertex AI Workbench (CPU) — `e2-standard-4` 정도의 인스턴스에서 실행. Colab Enterprise로도 대체 가능

---

## 6단계 — PyTorch 딥러닝: LSTM · CNN · CNN-LSTM

`LstmStockPyTorch.py`, `TimeSeriesWindow.py`, `CnnTimeSeriesFeature.py`, `CnnCandleChart.py`, `CnnLstmHybrid.py`에 해당합니다. 여기서부터 GPU 학습이 의미를 가집니다.

**Local Python**
```python
class LSTMNet(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=1, hidden_size=64)
        self.fc = nn.Linear(64, 1)
opt = torch.optim.Adam(model.parameters())
```

**AWS — SageMaker PyTorch Estimator + Deep Learning Container**
- 기존 `train()` 함수를 `entry_point` 스크립트로 그대로 옮김
- GPU: `ml.g5.xlarge` 등, 다중 GPU는 Distributed Data Parallel 라이브러리 사용
- 모델은 S3에, 학습로그는 CloudWatch로 자동 수집
```python
from sagemaker.pytorch import PyTorch
est = PyTorch(entry_point="lstm_train.py", role=role,
    framework_version="2.2", py_version="py310",
    instance_type="ml.g5.xlarge", instance_count=1,
    hyperparameters={"seq_len": 20, "epochs": 50})
est.fit({"train": "s3://bkt/train"})
```

**GCP — Vertex AI Training (PyTorch 사전빌드 컨테이너)**
- 사전빌드 PyTorch 컨테이너에 학습 스크립트만 패키징해 제출
- GPU: `n1-standard-8` + `NVIDIA_TESLA_T4`, 또는 TPU도 선택 가능
- 결과물은 Cloud Storage, 로그는 Cloud Logging
```bash
gcloud ai custom-jobs create --region=us-central1 \
  --display-name=lstm-train \
  --worker-pool-spec=machine-type=n1-standard-8,\
replica-count=1,accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2:latest,\
local-package-path=./lstm,script=lstm_train.py \
  --args=--seq_len=20,--epochs=50
```

> `TimeSeriesWindow.py`처럼 CLI 인자(`--ticker --window --horizon`)를 받는 구조는 두 플랫폼 모두 `hyperparameters`/`--args`로 그대로 전달하면 되므로 코드 수정이 거의 필요 없습니다.

---

## 7단계 — Transformer: Multi-Head Self-Attention

`TransformerAttention.py`에 해당합니다. 구조 자체는 6단계와 같은 PyTorch 학습 경로를 쓰되, 연산량이 늘어나므로 분산·가속 옵션이 중요해집니다.

- **Local Python**: `torch.nn.MultiheadAttention`으로 Q/K/V 프로젝션과 Attention 가중치를 직접 계산해 heatmap으로 시각화
- **AWS**: SageMaker Distributed Data Parallel(SMDDP)로 다중 GPU 스케일아웃. 대형 모델은 SageMaker Model Parallel, 추론 비용 절감은 Inferentia(Inf2)
- **GCP**: Attention 연산에 최적화된 Cloud TPU v5e를 Vertex AI Training에서 바로 사용 가능. 다중 GPU는 Reduction Server 또는 JAX/PyTorch XLA 분산

---

## 추가 — Autoencoder 이상탐지

`AutoencoderAnomalyDetect.py`에 해당합니다.

- **Local Python**: `torch.nn` Encoder-Decoder. 재구성 오차(reconstruction error)가 큰 구간을 이상치로 표시
- **AWS**: 시계열 이상탐지 전용 내장 알고리즘인 **Random Cut Forest(RCF)**가 개념적으로 가장 가까우며, 재구성 오차 기반 오토인코더를 그대로 쓰려면 PyTorch Script Mode 사용
- **GCP**: **BigQuery ML `AUTOENCODER`**로 SQL만으로 학습·이상탐지 스코어링까지 가능
```sql
CREATE OR REPLACE MODEL `proj.ds.ae_anomaly`
OPTIONS(model_type='AUTOENCODER', hidden_units=[32, 16, 32])
AS SELECT * FROM `proj.ds.features`;

SELECT * FROM ML.DETECT_ANOMALIES(
  MODEL `proj.ds.ae_anomaly`, STRUCT(0.02 AS contamination));
```

---

## 추가 — DQN 강화학습 트레이딩 에이전트

`DqnTradingAgent.py`에 해당합니다. Replay buffer(`deque`) + epsilon-greedy + Q-network 구조로, 두 클라우드 모두 전용 관리형 RL 서비스보다는 **Ray RLlib을 학습 인프라 위에 올리는 방식**이 현재 권장 경로입니다.

- **Local Python**: 직접 구현한 환경(주가 시뮬레이터)과 Q-network를 한 프로세스에서 학습
- **AWS**: SageMaker의 전용 RL 컨테이너는 단종 수순이라, 현재는 **Ray RLlib**을 SageMaker Training Job 컨테이너 위에서 직접 실행하는 방식이 표준. 환경은 Gym 인터페이스로 감싸 재사용
```python
est = PyTorch(entry_point="rllib_train.py",
    framework_version="2.2", instance_type="ml.g5.xlarge",
    dependencies=["requirements.txt"])  # ray[rllib] 포함
est.fit()
```
- **GCP**: Vertex AI가 관리형 **Ray 클러스터**를 네이티브 지원하므로, RLlib 학습 스크립트를 Vertex AI Training에 그대로 제출 가능 (**Ray on Vertex AI**)
```bash
gcloud ai persistent-resources create ray-cluster \
  --region=us-central1 --resource-pool-spec=... \
  --ray-metadata=...
```

---

## 추가 — 웹앱 서빙: Flask 백엔드

`webapp/backend/app.py` (Flask + flask-cors)에 해당합니다.

**Local Python**
```python
app = Flask(__name__)
CORS(app)

@app.route("/api/predict")
def predict(): ...
```

**AWS — App Runner / SageMaker Endpoint**
- Flask 앱 자체를 배포하려면 **App Runner** 또는 ECS Fargate (컨테이너 그대로)
- 추론 API만 분리하려면 학습된 모델을 **SageMaker Real-time Endpoint**로 배포하고 Flask는 프록시 역할만 수행
```bash
aws apprunner create-service --service-name stock-webapp \
  --source-configuration ImageRepository={...}
```

**GCP — Cloud Run**

Flask 앱을 Dockerfile 그대로 컨테이너화해 배포하면 요청이 없을 때 0으로 스케일되는 서버리스 운영이 가능합니다. 모델은 Cloud Storage에서 불러오거나 Vertex AI Endpoint를 호출합니다.
```bash
gcloud run deploy stock-webapp \
  --source=./webapp/backend --region=us-central1 \
  --allow-unauthenticated
```

---

## 종합 비교

동일 작업이라도 AWS와 GCP는 "관리형의 결"이 다릅니다. AWS는 SageMaker라는 단일 플랫폼 안에서 알고리즘별 세부 서비스가 잘게 나뉘어 있고, GCP는 BigQuery ML로 SQL 기반 학습 상당수를 흡수하는 대신 Vertex AI가 커스텀 학습을 담당합니다.

| 항목 | AWS | GCP |
|------|-----|-----|
| 중심 플랫폼 | Amazon SageMaker (통합 스위트) | Vertex AI + BigQuery ML |
| SQL 기반 ML | 제한적 (Athena/Redshift ML) | BigQuery ML이 회귀/분류/군집/PCA/시계열/오토인코더까지 폭넓게 지원 |
| 강점 | 알고리즘별 세분화된 내장 옵션, 성숙한 MLOps(Pipelines) | 데이터 웨어하우스와 ML의 밀접한 통합, TPU |
| 가속기 | NVIDIA GPU, Trainium/Inferentia | NVIDIA GPU, Cloud TPU |
| 과금 단위 | 인스턴스 시간 + 스토리지 + 데이터 전송 | 컴퓨트 시간 + BigQuery 쿼리 처리량 |
| 공통 | 관리형 Jupyter 노트북, 온디맨드 GPU/TPU, REST 서빙 엔드포인트, 실험 로그/메트릭 자동 수집 | (좌동) |

---

## 추천 경로

이 커리큘럼을 그대로 따라가며 클라우드로 확장한다면, 규모에 따라 아래 순서를 권장합니다.

1. **개인 학습/포트폴리오 단계** — 로컬 venv 그대로 진행하고, 결과 공유용 노트북만 SageMaker Studio Lab(무료) 또는 Vertex AI Workbench / Colab으로 옮기기.
2. **데이터 규모가 커질 때(다종목·장기간)** — 수집·정규화를 GCP는 BigQuery, AWS는 S3+Glue+Athena로 이전. sklearn/statsmodels 코드는 거의 그대로 유지.
3. **PyTorch 딥러닝(6~7단계) GPU가 필요할 때** — SageMaker Training Job 또는 Vertex AI Custom Job으로 `entry_point`만 감싸서 제출. 로컬 `train()` 함수 구조를 바꿀 필요 없음.
4. **서비스화(webapp)** — 트래픽이 예측 불가능하면 Cloud Run(GCP)이 콜드스타트·과금 면에서 유리하고, SageMaker 생태계에 이미 있다면 App Runner+SageMaker Endpoint 조합(AWS) 유지.

---

> 서비스명·API·요금 체계는 각 클라우드가 자주 개편하므로, 실제 적용 전에 AWS/GCP 공식 문서에서 최신 사양을 재확인하세요.
