import os
import time
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 65)
print("  하이퍼파라미터 튜닝 & 검증 전략 실습")
print("=" * 65)
print()
print("  핵심 개념 비유:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  데이터 분할  = 시험지를 미리 보지 않는 규칙         │")
print("  │  K-Fold CV   = 5번 돌아가며 시험을 보는 것           │")
print("  │  Grid Search = 모든 레시피를 직접 만들어 최고 선택   │")
print("  │  과적합      = 족보만 외워 실전에서 틀리는 것        │")
print("  │  Dropout/정규화 = '한 방법만 믿지 마' 라는 규칙      │")
print("  └──────────────────────────────────────────────────────┘")

np.random.seed(42)

# ── 1. 데이터 생성 ─────────────────────────────────────────
print("\n[1/8] 가상 주식 분류 데이터 생성 중 (400개 샘플, 8개 특징)...")
print("   특징: RSI, MACD, 볼린저밴드%, 거래량비, 이평선 기울기 등")
time.sleep(0.5)

N = 400
np.random.seed(42)
X_raw = np.random.randn(N, 8)
# 비선형 경계로 레이블 생성 (실제 주식처럼 완벽히 분리되지 않음)
y = ((X_raw[:, 0] + X_raw[:, 2] * 0.5 - X_raw[:, 4] * 0.3
      + np.random.normal(0, 0.4, N)) > 0).astype(int)
print(f"   → {N}개 샘플  |  상승: {y.sum()}개  하락: {(y==0).sum()}개")
time.sleep(0.3)

# ── 2. 데이터 분할 전략 비교 ──────────────────────────────
print("\n[2/8] 데이터 분할 전략 비교 중...")
print("   ① Hold-out (7:1.5:1.5) — 빠르지만 운에 의존")
print("   ② K-Fold CV (k=5)      — 5번 평균으로 신뢰도 높음")
time.sleep(0.5)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

print(f"   Hold-out → 학습: {len(X_train)}  검증: {len(X_val)}  테스트: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
X_tv_s    = scaler.fit_transform(X_trainval)
time.sleep(0.3)

# ── 3. K-Fold 교차검증 ────────────────────────────────────
print("\n[3/8] K-Fold 교차검증 (k=5) 실행 중...")
time.sleep(0.4)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, random_state=42)),
])
cv_scores = cross_val_score(base_model, X_trainval, y_trainval, cv=kf, scoring='accuracy')
print(f"   Fold별 정확도: {[f'{s:.3f}' for s in cv_scores]}")
print(f"   평균 정확도 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
time.sleep(0.3)

# ── 4. Grid Search 하이퍼파라미터 튜닝 ────────────────────
print("\n[4/8] Grid Search로 SVM 하이퍼파라미터 탐색 중...")
print("   C (정규화 강도)  × kernel 조합을 모두 검증합니다")
time.sleep(0.5)

param_grid = {
    'clf__C':      [0.01, 0.1, 1.0, 10.0],
    'clf__kernel': ['linear', 'rbf'],
}
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True, random_state=42)),
])
grid_search = GridSearchCV(svm_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_trainval, y_trainval)

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
print(f"   최적 파라미터: {best_params}")
print(f"   CV 최고 정확도: {best_cv_score:.4f}")
time.sleep(0.3)

# ── 5. 과적합 실험 ────────────────────────────────────────
print("\n[5/8] 과적합 실험 중 (정규화 강도 C 변화)...")
print("   C가 너무 크면 훈련 정확도↑, 검증 정확도↓ → 과적합!")
time.sleep(0.5)

C_values  = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accs, val_accs = [], []
for C in C_values:
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=C, kernel='rbf', random_state=42)),
    ])
    svm.fit(X_train, y_train)
    train_accs.append(svm.score(X_train_s, y_train))
    val_accs.append(svm.score(X_val_s, y_val))
    print(f"   C={C:6.3f} | 학습={train_accs[-1]:.3f}  검증={val_accs[-1]:.3f}"
          f"  {'⚠ 과적합' if train_accs[-1] - val_accs[-1] > 0.08 else ''}")
    time.sleep(0.1)
time.sleep(0.2)

# ── 6. Dropout 효과 (PyTorch MLP) ─────────────────────────
print("\n[6/8] Dropout 과적합 방지 효과 실험 중...")
print("   Dropout 없음 vs Dropout(0.5) 비교")
time.sleep(0.5)

import torch
import torch.nn as nn

torch.manual_seed(42)
X_tr = torch.tensor(X_train_s, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_v  = torch.tensor(X_val_s,  dtype=torch.float32)
y_v  = torch.tensor(y_val,    dtype=torch.long)

def make_mlp(dropout=0.0):
    layers = [nn.Linear(8, 64), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(64, 64), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(64, 2))
    return nn.Sequential(*layers)

def train_model(model, epochs=150):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    tr_hist, val_hist = [], []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            tr_hist.append((model(X_tr).argmax(1) == y_tr).float().mean().item())
            val_hist.append((model(X_v).argmax(1) == y_v).float().mean().item())
    return tr_hist, val_hist

model_no_drop = make_mlp(0.0)
model_dropout = make_mlp(0.5)
tr_no, val_no = train_model(model_no_drop)
tr_do, val_do = train_model(model_dropout)
print(f"   Dropout 없음 → 학습={tr_no[-1]:.3f}  검증={val_no[-1]:.3f}")
print(f"   Dropout 0.5  → 학습={tr_do[-1]:.3f}  검증={val_do[-1]:.3f}")
time.sleep(0.3)

# ── 7. 최종 테스트 평가 ────────────────────────────────────
print("\n[7/8] 최종 테스트 평가 (최적 SVM)...")
time.sleep(0.4)
best_model = grid_search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"   → 최적 파라미터: {best_params}")
print(f"   → 테스트 정확도: {test_acc:.4f} ({test_acc*100:.1f}%)")
time.sleep(0.3)

# ── 8. 시각화 ─────────────────────────────────────────────
print("\n[8/8] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(14, 10))

# Grid Search 결과 히트맵
ax1 = fig.add_subplot(2, 2, 1)
results = grid_search.cv_results_
scores = results['mean_test_score'].reshape(4, 2)
im = ax1.imshow(scores, cmap='RdYlGn', vmin=0.4, vmax=0.9, aspect='auto')
plt.colorbar(im, ax=ax1, fraction=0.046)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['linear', 'rbf'])
ax1.set_yticks(range(4))
ax1.set_yticklabels(['0.01', '0.1', '1.0', '10.0'])
ax1.set_xlabel("kernel")
ax1.set_ylabel("C")
ax1.set_title("Grid Search CV 정확도 히트맵")
for i, j in itertools.product(range(4), range(2)):
    ax1.text(j, i, f"{scores[i, j]:.3f}", ha='center', va='center', fontsize=9,
             color='white' if scores[i, j] < 0.6 else 'black')

# 과적합 실험 (C 변화)
ax2 = fig.add_subplot(2, 2, 2)
ax2.semilogx(C_values, train_accs, 'o-', color='tomato', label='학습 정확도')
ax2.semilogx(C_values, val_accs,   's-', color='steelblue', label='검증 정확도')
ax2.set_title("C 값에 따른 과적합 변화")
ax2.set_xlabel("C (정규화 강도, log 스케일)")
ax2.set_ylabel("정확도")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.axvspan(100, 1200, alpha=0.1, color='red', label='과적합 위험')

# Dropout 효과
ax3 = fig.add_subplot(2, 2, 3)
epochs_range = range(1, 151)
ax3.plot(epochs_range, val_no, color='tomato', linewidth=1.5, label='Dropout 없음 (검증)')
ax3.plot(epochs_range, val_do, color='steelblue', linewidth=1.5, label='Dropout 0.5 (검증)')
ax3.plot(epochs_range, tr_no, '--', color='tomato', linewidth=0.8, alpha=0.5, label='Dropout 없음 (학습)')
ax3.plot(epochs_range, tr_do, '--', color='steelblue', linewidth=0.8, alpha=0.5, label='Dropout 0.5 (학습)')
ax3.set_title("Dropout 과적합 방지 효과")
ax3.set_xlabel("에폭")
ax3.set_ylabel("정확도")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# K-Fold 교차검증 결과
ax4 = fig.add_subplot(2, 2, 4)
folds = [f"Fold {i+1}" for i in range(5)]
bars = ax4.bar(folds, cv_scores, color='mediumseagreen', alpha=0.8, edgecolor='k', linewidth=0.5)
ax4.axhline(cv_scores.mean(), linestyle='--', color='red', linewidth=1.5,
            label=f'평균={cv_scores.mean():.3f}')
ax4.set_ylim(0.4, 1.0)
ax4.set_title(f"K-Fold(k=5) 교차검증 정확도  |  테스트={test_acc:.3f}")
ax4.set_ylabel("정확도")
ax4.legend()
ax4.grid(alpha=0.3, axis='y')
for bar, score in zip(bars, cv_scores):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{score:.3f}", ha='center', va='bottom', fontsize=9)

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         "파라미터 조합을 전부 시험해보고, 5번 교차검증으로 결과를 믿을 수 있게 확인합니다",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── Grid Search 히트맵 (ax1) ──
# 가장 밝은 칸 위치(최고점) 찾기
best_i, best_j = np.unravel_index(scores.argmax(), scores.shape)
ax1.annotate('최적 조합이에요!', xy=(best_j, best_i), xytext=(best_j + 0.5, best_i - 0.8),
             xycoords='data', textcoords='data',
             arrowprops=dict(arrowstyle='->', color='#333'), fontsize=7, color='#333')
# 가장 어두운 칸 위치(최저점)
worst_i, worst_j = np.unravel_index(scores.argmin(), scores.shape)
ax1.text(worst_j, worst_i + 0.35, '이 조합은\n별로예요', ha='center', va='center',
         fontsize=6.5, color='#333',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

# ── 과적합 패널 (ax2) ──
ax2.annotate('여기서부터 과적합\n— 시험 문제만 외운 상태', xy=(100, train_accs[-1]),
             xytext=(10, train_accs[-1] - 0.08),
             xycoords='data', textcoords='data',
             arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=7, color='tomato')
ax2.text(0.18, 0.35, '이 구간이 적당해요',
         transform=ax2.transAxes, fontsize=7, color='#333',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
ax2.text(0.80, 0.18, 'C가 너무 크면\n훈련은 100점, 실전은 꽝',
         transform=ax2.transAxes, fontsize=7, color='tomato', ha='center')

# ── Dropout 패널 (ax3) ──
ax3.annotate('과적합 — 외운 것만 잘해요', xy=(0.65, 0.82), xytext=(0.30, 0.90),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=7, color='tomato')
ax3.annotate('Dropout으로 과적합 방지!', xy=(0.80, 0.60), xytext=(0.40, 0.45),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='steelblue'), fontsize=7, color='steelblue')

# ── K-Fold 패널 (ax4) ──
ax4.text(0.5, -0.14,
         '5번 다른 시험을 봐서 평균냈어요 — 한 번보다 훨씬 믿을 수 있어요',
         transform=ax4.transAxes, ha='center', fontsize=7, color='gray')
ax4.annotate('이 정도면 신뢰할 수 있어요', xy=(0.5, cv_scores.mean()),
             xytext=(0.55, cv_scores.mean() - 0.07),
             xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=7, color='red')

plt.subplots_adjust(top=0.93)
plt.tight_layout()
plt.savefig("../result/HyperparamTuning.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/HyperparamTuning.png")

print("\n✓ 하이퍼파라미터 튜닝 & 교차검증 & 과적합 방지 실습 완료!\n")
