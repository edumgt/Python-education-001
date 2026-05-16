import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  Word2Vec: 투자 뉴스 단어 임베딩 실습")
print("=" * 55)

print("\n[1/5] 투자 뉴스 토큰 데이터 로딩 중...")
time.sleep(0.5)
sentences = [
    ["금리", "인하", "기대", "성장주", "상승"],
    ["반도체", "실적", "개선", "외국인", "순매수"],
    ["배당", "확대", "금융주", "투자심리", "개선"],
    ["원자재", "상승", "제조업", "마진", "악화"],
    ["달러", "강세", "수출주", "실적", "기대"],
    ["경기", "둔화", "방어주", "관심", "확대"],
]
for i, sent in enumerate(sentences, 1):
    print(f"   문장{i}: {' / '.join(sent)}")
    time.sleep(0.2)

print("\n[2/5] Word2Vec 모델 학습 중...")
print("   원리: 주변 단어(window=3)로 가운데 단어 예측 훈련")
print("   결과: 비슷한 문맥 단어 → 벡터 공간에서 가까운 위치")
time.sleep(0.8)
model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4,
)
print(f"   → 학습 완료!  어휘 크기: {len(model.wv)}개 단어")
time.sleep(0.5)

print("\n[3/5] 학습된 어휘 목록:")
time.sleep(0.3)
for i, word in enumerate(model.wv.index_to_key):
    print(f"   [{i + 1:2d}] {word}", end="  ")
    if (i + 1) % 5 == 0:
        print()
    time.sleep(0.08)
print()

print("\n[4/5] '실적' 단어의 50차원 임베딩 벡터 확인...")
time.sleep(0.5)
target_word = "실적"
if target_word in model.wv:
    vec = model.wv[target_word]
    print(f"   벡터 앞 10차원: {vec[:10].round(4)}")
    print(f"   (총 {len(vec)}차원 벡터로 단어 의미를 수치화)")
time.sleep(0.5)

print("\n[5/5] '실적'과 가장 유사한 단어 찾기 (코사인 유사도)...")
time.sleep(0.5)
if target_word in model.wv:
    similar = model.wv.most_similar(target_word, topn=5)
    print(f"   기준 단어: '{target_word}'")
    for word, score in similar:
        bar = "█" * int(score * 10)
        print(f"   {word:8s}  유사도: {score:.4f}  {bar}")
        time.sleep(0.2)

print("\n[시각화] 단어 벡터를 2D로 축소하여 산점도 그리는 중 (PCA)...")
time.sleep(0.4)

# 모든 단어 벡터를 2D로 축소
words = model.wv.index_to_key
vectors = np.array([model.wv[w] for w in words])
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(coords[:, 0], coords[:, 1], s=60, color='steelblue',
           alpha=0.8, edgecolors='k', linewidths=0.4, zorder=2)

# 단어 이름 표시
for word, (x, y) in zip(words, coords):
    ax.text(x + 0.01, y + 0.01, word, fontsize=8, color='#333')

ax.set_title("Word2Vec 단어 임베딩 2D 지도")

# ── 어노테이션: 그래프 상단 한 줄 요약 ──────────────────────────────────
fig.text(0.5, 0.98,
         "비슷한 상황에서 나오는 단어끼리 가깝게 배치합니다 — '반도체'와 '삼성'은 가까이, '배당'과 '기술주'는 멀리",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 어노테이션: "점 하나 = 단어 하나" 설명 ──────────────────────────────
# 임의의 한 점에 화살표로 안내
ref_idx = 0
rx, ry = coords[ref_idx]
ax.annotate("점 하나 = 단어 하나",
            xy=(rx, ry),
            xytext=(rx + 0.15, ry + 0.15),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=7, color='#333')

# ── 어노테이션: 가장 가까운 단어 쌍 사이 선과 텍스트 ────────────────────
# 코사인 유사도 기준 가장 유사한 쌍 찾기
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(vectors)
np.fill_diagonal(sim_matrix, -1)
close_i, close_j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
cx1, cy1 = coords[close_i]
cx2, cy2 = coords[close_j]
ax.plot([cx1, cx2], [cy1, cy2], color='tomato', linewidth=1.2,
        linestyle='--', alpha=0.7, zorder=1)
mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
ax.text(mx, my + 0.03,
        "자주 같이 나오는\n단어들은 가까워요",
        fontsize=7, color='tomato', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='tomato'))

# ── 어노테이션: 가장 먼 단어 쌍 사이 선과 텍스트 ────────────────────────
sim_matrix2 = cosine_similarity(vectors)
np.fill_diagonal(sim_matrix2, 2)
far_i, far_j = np.unravel_index(sim_matrix2.argmin(), sim_matrix2.shape)
fx1, fy1 = coords[far_i]
fx2, fy2 = coords[far_j]
ax.plot([fx1, fx2], [fy1, fy2], color='slategray', linewidth=0.8,
        linestyle=':', alpha=0.5, zorder=1)
fmx, fmy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
ax.text(fmx, fmy - 0.05,
        "관련 없는 단어는\n멀리 있어요",
        fontsize=7, color='slategray', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='lightgray'))

# ── 어노테이션: x축·y축 보충 설명 ───────────────────────────────────────
ax.text(0.5, -0.08,
        "PCA로 50차원 벡터를 2개의 숫자로 줄여 화면에 표시한 것이에요",
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("result/Word2VecEmbedding.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/Word2VecEmbedding.png")

print("\n✓ Word2Vec 임베딩 실습 완료!\n")
