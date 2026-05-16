import time

from gensim.models import Word2Vec

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

print("\n✓ Word2Vec 임베딩 실습 완료!\n")
