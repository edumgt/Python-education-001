from gensim.models import Word2Vec

# 예제 데이터
sentences = [
    ["나는", "인공지능", "공부하고", "있다"],
    ["인공지능", "자연어", "처리", "관련이", "있다"],
    ["인공지능", "머신러닝", "밀접한", "관계가", "있다"],
    ["인공지능", "산업에서", "활용된다"],
    ["인공지능", "미래", "기술의", "핵심이다"],
    ["인공지능", "중요한", "기술이다"],
    ["인간지능", "중요한", "기술이다"],
    ["기계지능", "중요한", "기술이다"]
]

# Word2Vec 모델 학습
model = Word2Vec(sentences=sentences, vector_size=100,
                 window=5, min_count=1, workers=4)

# 학습된 단어 목록 확인
print("학습된 단어 목록:", model.wv.index_to_key)

# 특정 단어 벡터 출력
if "인공지능" in model.wv:
    print("인공지능 벡터:", model.wv["인공지능"])
else:
    print("인공지능 단어가 포함되지 않았습니다.")

# 유사한 단어 찾기
if "인공지능" in model.wv:
    print("인공지능과 유사한 단어:", model.wv.most_similar("인공지능"))
else:
    print("유사한 단어를 찾을 수 없습니다.")
