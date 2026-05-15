from sklearn.feature_extraction.text import TfidfVectorizer

# 문서 리스트
documents = [
    "나는 인공지능을 공부하고 있다",
    "인공지능은 미래의 핵심 기술이다",
    "인공지능은 우리의 핵심 기술이다",
    "인공지능 기술을 선도하는 기업",
    "자연어 처리는 인공지능의 한 분야다"
]

# TF-IDF 벡터로 특징 추출
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 결과 출력
print("단어 목록:", vectorizer.get_feature_names_out())
print("TF-IDF 행렬:\n", tfidf_matrix.toarray())
