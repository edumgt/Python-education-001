import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  TF-IDF: 주식 뉴스 문장 → 수치 벡터 변환 실습")
print("=" * 55)

print("\n[1/4] 투자 뉴스 문장 로딩 중...")
time.sleep(0.5)
documents = [
    "금리 인하 기대감에 성장주가 반등했다",
    "반도체 수요 회복으로 실적 전망이 상향됐다",
    "원자재 가격 상승이 제조업 수익성에 부담을 줬다",
    "외국인 순매수 확대에 코스피가 상승 마감했다",
    "배당 확대 발표로 금융주 투자심리가 개선됐다",
]
for i, doc in enumerate(documents, 1):
    print(f"   문서{i}: {doc}")
    time.sleep(0.2)

print("\n[2/4] TF-IDF 벡터라이저 학습 중 (어휘 사전 구축)...")
print("   TF  = 단어가 해당 문서 안에서 얼마나 자주 나오는가")
print("   IDF = 전체 문서 중 해당 단어가 등장하는 문서 수의 역수")
print("   TF-IDF 높음 = 이 문서에서만 자주 쓰이는 핵심 단어")
time.sleep(0.8)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()
print(f"   → 어휘 사전 크기: {len(vocab)}개 단어")
time.sleep(0.3)

print("\n[3/4] 학습된 어휘 사전 단어 목록:")
time.sleep(0.3)
for i, word in enumerate(vocab):
    print(f"   [{i:2d}] {word}", end="  ")
    if (i + 1) % 5 == 0:
        print()
    time.sleep(0.06)
print()

print("\n[4/4] 문서별 TF-IDF 핵심 단어 추출 중...")
time.sleep(0.5)
matrix = tfidf_matrix.toarray()
print(f"   행렬 크기: {matrix.shape[0]}문서 × {matrix.shape[1]}단어")
for i, row in enumerate(matrix):
    top_idx = row.argsort()[::-1][:3]
    top = [(vocab[j], round(row[j], 3)) for j in top_idx if row[j] > 0]
    print(f"   문서{i + 1} 핵심 단어(TF-IDF 상위3): {top}")
    time.sleep(0.2)

print("\n[시각화] 문서1의 TF-IDF 점수 막대 그래프 생성 중...")
time.sleep(0.4)
# 문서1의 TF-IDF 점수를 내림차순 정렬
doc_idx = 0
scores = matrix[doc_idx]
sorted_indices = scores.argsort()[::-1]
sorted_words = [vocab[i] for i in sorted_indices]
sorted_scores = scores[sorted_indices]

# 0점 단어 제외 (등장하지 않은 단어)
nonzero_mask = sorted_scores > 0
sorted_words = [w for w, nz in zip(sorted_words, nonzero_mask) if nz]
sorted_scores = sorted_scores[nonzero_mask]

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ['tomato' if i == 0 else 'steelblue' for i in range(len(sorted_words))]
bars = ax.bar(range(len(sorted_words)), sorted_scores, color=bar_colors, alpha=0.8, edgecolor='k', linewidth=0.4)
ax.set_xticks(range(len(sorted_words)))
ax.set_xticklabels(sorted_words, rotation=30, ha='right', fontsize=8)
ax.set_title("TF-IDF 단어 중요도 (문서1: 금리 인하 기대감 뉴스)")

# ── 어노테이션: 그래프 상단 한 줄 요약 ──────────────────────────────────
fig.text(0.5, 0.98,
         "뉴스에서 자주 나오면서도 특별한 단어에 높은 점수를 줍니다 (TF-IDF)",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 어노테이션: 가장 높은 막대에 화살표 ─────────────────────────────────
ax.annotate("이 단어가 가장\n중요한 단어예요",
            xy=(0, sorted_scores[0]),
            xytext=(1.5, sorted_scores[0] * 0.9),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=7, color='#333')

# ── 어노테이션: x축 보충 설명 ───────────────────────────────────────────
ax.text(0.5, -0.28,
        "뉴스에 나온 단어들",
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')

# ── 어노테이션: y축 보충 설명 ───────────────────────────────────────────
ax.text(-0.14, 0.5,
        "중요도 점수 (높을수록 이 뉴스에서 특별한 단어)",
        transform=ax.transAxes, ha='center', fontsize=7, color='gray',
        rotation=90, va='center')

# ── 어노테이션: 구역 설명 텍스트 ────────────────────────────────────────
ax.text(len(sorted_words) - 1, sorted_scores[0] * 0.5,
        "점수가 낮은 단어는\n다른 뉴스에도 흔히 나와요",
        fontsize=7, color='gray', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='lightgray'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("result/TfidfNewsFeature.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/TfidfNewsFeature.png")

print("\n✓ TF-IDF 특징 추출 실습 완료!\n")
