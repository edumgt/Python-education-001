# HyperparamTuning.py 코드 설명

> 한 줄 요약: 주식 데이터를 예로 들어, 인공지능 모델의 "설정값(하이퍼파라미터)"을 가장 좋게 맞추는 방법과 "과적합"을 막는 방법을 실험하는 코드입니다.

---

## 이 코드가 하는 일

요리할 때 설탕을 얼마나 넣어야 맛있는지 여러 번 직접 만들어 봐야 알 수 있죠? 인공지능 모델도 마찬가지예요. "설탕 양"처럼 사람이 미리 정해줘야 하는 값을 **하이퍼파라미터**라고 해요. 이 코드는 가상의 주식 데이터를 만들어서, 여러 설정값 조합을 모두 시험해 보고 가장 정확한 조합을 자동으로 찾아줍니다.

또 "족보(시험 문제)만 외워서 실전 시험에서 틀리는 것"처럼, 모델이 배운 데이터에만 너무 딱 맞아버려서 새 데이터에는 엉터리가 되는 **과적합** 현상도 직접 보여줍니다. 그리고 이것을 막는 **Dropout** 기법도 실험합니다.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `os` | 결과물을 저장할 폴더를 만들 때 사용해요 |
| `time` | 단계마다 잠깐 기다려서 출력을 읽기 쉽게 해줘요 |
| `itertools` | 여러 값의 모든 조합을 만들 때 쓰는 도구예요 |
| `matplotlib.pyplot` | 그래프를 그려주는 도구예요 |
| `numpy` | 숫자 계산을 빠르게 해주는 도구예요 |
| `sklearn.model_selection` | 데이터 나누기, 교차검증, Grid Search 등 모델 평가 도구 모음이에요 |
| `sklearn.svm.SVC` | SVM이라는 인공지능 분류 모델이에요 |
| `sklearn.linear_model.LogisticRegression` | 확률로 분류하는 로지스틱 회귀 모델이에요 |
| `sklearn.preprocessing.StandardScaler` | 데이터의 크기를 통일시켜 주는 도구예요 |
| `sklearn.pipeline.Pipeline` | 데이터 전처리와 모델 학습을 한 줄로 묶어주는 도구예요 |
| `korean_font` | 그래프에 한국어가 깨지지 않게 해주는 설정이에요 |
| `torch` / `torch.nn` | 딥러닝 신경망 모델을 만드는 도구예요 (PyTorch) |

---

## 코드 흐름 (단계별 설명)

### 1단계: 가상 주식 데이터 만들기

```python
N = 400
X_raw = np.random.randn(N, 8)
y = ((X_raw[:, 0] + X_raw[:, 2] * 0.5 - X_raw[:, 4] * 0.3
      + np.random.normal(0, 0.4, N)) > 0).astype(int)
```

> 📌 **쉬운 설명:** RSI, MACD 등 주식 관련 숫자 8개를 가진 400개의 가짜 주식 데이터를 만들어요. 그리고 주가가 "올랐는지(1)" "내렸는지(0)" 를 정답으로 붙여줘요. 실제 주식처럼 완벽히 나뉘지 않고 조금 섞여 있어요.

---

### 2단계: 데이터 나누기

```python
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42
)
```

> 📌 **쉬운 설명:** 400개의 데이터를 "공부할 문제(학습)", "중간 확인 문제(검증)", "최종 시험 문제(테스트)" 세 묶음으로 나눠요. 시험지를 미리 보면 안 되듯이, 모델이 테스트 데이터를 학습 때 보면 안 돼요.

---

### 3단계: K-Fold 교차검증

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(base_model, X_trainval, y_trainval, cv=kf, scoring='accuracy')
print(f"평균 정확도: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

> 📌 **쉬운 설명:** 데이터를 5조각으로 나눠서, 한 조각씩 번갈아 시험지로 쓰고 나머지로 공부해요. 이렇게 5번 시험을 보고 평균을 내면 운에 덜 좌우되고 훨씬 믿을 수 있는 점수가 나와요. 예를 들어 국어, 수학, 영어, 과학, 사회를 번갈아 시험 보는 것처럼요!

---

### 4단계: Grid Search로 최적 설정값 찾기

```python
param_grid = {
    'clf__C':      [0.01, 0.1, 1.0, 10.0],
    'clf__kernel': ['linear', 'rbf'],
}
grid_search = GridSearchCV(svm_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_trainval, y_trainval)
```

> 📌 **쉬운 설명:** `C` 값 4가지와 `kernel` 2가지를 곱하면 8가지 조합이 생겨요. 이 8가지 레시피를 모두 직접 요리해 보고, 가장 맛있는(정확도 높은) 레시피를 골라줘요. 이것이 **Grid Search**예요!

---

### 5단계: 과적합 실험

```python
for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    svm.fit(X_train, y_train)
    train_accs.append(svm.score(X_train_s, y_train))
    val_accs.append(svm.score(X_val_s, y_val))
```

> 📌 **쉬운 설명:** `C` 값을 점점 크게 바꿔가며 "공부할 때 점수"와 "시험 점수"를 비교해요. `C`가 너무 크면 공부할 때는 100점이지만 시험에서는 꽝이에요. 이게 바로 **과적합**이에요!

---

### 6단계: Dropout으로 과적합 막기

```python
def make_mlp(dropout=0.0):
    layers = [nn.Linear(8, 64), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    ...
```

> 📌 **쉬운 설명:** 신경망을 학습할 때 랜덤으로 일부 뉴런을 껐다 켰다 해요. 이렇게 하면 특정 경로에만 의존하지 않게 되어서 새 데이터에도 잘 작동해요. 공부할 때 항상 한 문제집만 풀지 않고 여러 문제집을 섞어 공부하는 것과 같아요.

---

### 7단계: 그래프로 결과 저장

```python
plt.savefig("../result/HyperparamTuning.png", dpi=150, bbox_inches="tight")
```

> 📌 **쉬운 설명:** 4가지 실험 결과(Grid Search 히트맵, 과적합 변화 그래프, Dropout 효과, K-Fold 결과)를 한 장의 그림으로 저장해요.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| 하이퍼파라미터 | 모델을 학습시키기 전에 사람이 먼저 정해줘야 하는 설정값이에요. 요리로 치면 소금을 얼마나 넣을지 같은 거예요. |
| K-Fold 교차검증 | 데이터를 K개의 묶음으로 나눠서 K번 번갈아 시험 보는 방법이에요. 더 믿을 수 있는 점수를 얻을 수 있어요. |
| Grid Search | 가능한 하이퍼파라미터 조합을 전부 시험해 보고 최고를 고르는 방법이에요. |
| 과적합 (Overfitting) | 공부한 문제만 잘 풀고 처음 보는 문제는 못 푸는 것이에요. 족보만 달달 외운 것과 같아요. |
| Dropout | 학습 중에 랜덤으로 일부 뉴런을 꺼서 과적합을 막는 기법이에요. |
| SVM (서포트 벡터 머신) | 두 그룹을 가장 넓은 경계선으로 나누는 분류 모델이에요. |
| C (정규화 강도) | SVM에서 경계선을 얼마나 엄격하게 그을지 결정하는 값이에요. 크면 훈련 데이터에 딱 맞지만 과적합 위험이 있어요. |
| Pipeline | 데이터 준비 → 모델 학습을 하나의 흐름으로 묶어주는 도구예요. |
| StandardScaler | 여러 숫자들의 크기를 통일해서 비교가 공정하게 되도록 만들어줘요. |
| 정확도 (Accuracy) | 전체 예측 중 맞게 예측한 비율이에요. 100개 중 80개 맞으면 80%예요. |

---

## 실행 결과

- **파일:** `result/HyperparamTuning.png`
- **내용:** 4개의 그래프가 한 장에 담긴 이미지
  - 왼쪽 위: Grid Search 결과 히트맵 — 어떤 `C`와 `kernel` 조합이 가장 좋은지 색깔로 보여줘요
  - 오른쪽 위: `C` 값을 바꿨을 때 과적합이 어떻게 생기는지 보여주는 그래프
  - 왼쪽 아래: Dropout을 쓸 때와 안 쓸 때 학습/검증 정확도 변화 그래프
  - 오른쪽 아래: K-Fold 5번 시험의 각 점수와 평균을 막대 그래프로 보여줘요
