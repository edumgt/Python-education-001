from gensim.models import Word2Vec

# 투자 뉴스 토큰 예시
sentences = [
    ["금리", "인하", "기대", "성장주", "상승"],
    ["반도체", "실적", "개선", "외국인", "순매수"],
    ["배당", "확대", "금융주", "투자심리", "개선"],
    ["원자재", "상승", "제조업", "마진", "악화"],
    ["달러", "강세", "수출주", "실적", "기대"],
    ["경기", "둔화", "방어주", "관심", "확대"],
]

model = Word2Vec(sentences=sentences, vector_size=50, window=3, min_count=1, workers=4)

print("학습된 단어 목록:", model.wv.index_to_key)

target_word = "실적"
if target_word in model.wv:
    print(f"'{target_word}' 벡터:", model.wv[target_word])
    print(f"'{target_word}'와 유사한 단어:", model.wv.most_similar(target_word))
else:
    print(f"'{target_word}' 단어가 포함되지 않았습니다.")
