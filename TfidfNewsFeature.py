from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트를 TF-IDF 수치 벡터로 변환하는 클래스

# 주식 뉴스 문장
documents = [
    "금리 인하 기대감에 성장주가 반등했다",       # 금리 정책 관련 뉴스
    "반도체 수요 회복으로 실적 전망이 상향됐다",   # 업종 실적 관련 뉴스
    "원자재 가격 상승이 제조업 수익성에 부담을 줬다",  # 비용 압력 관련 뉴스
    "외국인 순매수 확대에 코스피가 상승 마감했다",  # 수급 관련 뉴스
    "배당 확대 발표로 금융주 투자심리가 개선됐다",  # 배당 정책 관련 뉴스
]

vectorizer = TfidfVectorizer()                    # TF-IDF 벡터라이저 객체 생성 (기본 설정: 공백 기준 토크나이징)
tfidf_matrix = vectorizer.fit_transform(documents) # 문서 집합으로 어휘 학습(fit) + 각 문서를 TF-IDF 행렬로 변환(transform)
                                                   # 결과: 문서 수 × 어휘 수 크기의 희소 행렬(sparse matrix)

print("단어 목록:", vectorizer.get_feature_names_out())  # 학습된 어휘 사전의 단어 목록 출력 (열 이름에 해당)
print("TF-IDF 행렬:\n", tfidf_matrix.toarray())          # 희소 행렬을 일반 2D 배열로 변환하여 출력
                                                          # 각 셀 값: 해당 문서에서 해당 단어의 TF-IDF 점수
                                                          # TF(단어 빈도) × IDF(역문서 빈도)로 단어 중요도 표현
