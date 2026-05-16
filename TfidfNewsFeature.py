import time

from sklearn.feature_extraction.text import TfidfVectorizer

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

print("\n✓ TF-IDF 특징 추출 실습 완료!\n")
