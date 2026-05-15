from sklearn.feature_extraction.text import TfidfVectorizer

# 주식 뉴스 문장
documents = [
    "금리 인하 기대감에 성장주가 반등했다",
    "반도체 수요 회복으로 실적 전망이 상향됐다",
    "원자재 가격 상승이 제조업 수익성에 부담을 줬다",
    "외국인 순매수 확대에 코스피가 상승 마감했다",
    "배당 확대 발표로 금융주 투자심리가 개선됐다",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("단어 목록:", vectorizer.get_feature_names_out())
print("TF-IDF 행렬:\n", tfidf_matrix.toarray())
