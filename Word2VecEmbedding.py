from gensim.models import Word2Vec  # 단어 임베딩(Word2Vec) 모델 클래스

# 투자 뉴스 토큰 예시
sentences = [
    ["금리", "인하", "기대", "성장주", "상승"],          # 금리 정책 관련 문장 (이미 토크나이징 완료)
    ["반도체", "실적", "개선", "외국인", "순매수"],       # 반도체 업황 관련 문장
    ["배당", "확대", "금융주", "투자심리", "개선"],       # 배당 정책 관련 문장
    ["원자재", "상승", "제조업", "마진", "악화"],         # 원가 부담 관련 문장
    ["달러", "강세", "수출주", "실적", "기대"],           # 환율 영향 관련 문장
    ["경기", "둔화", "방어주", "관심", "확대"],           # 경기 방어 전략 관련 문장
]

model = Word2Vec(
    sentences=sentences,  # 학습할 토큰화된 문장 리스트
    vector_size=50,        # 각 단어를 표현할 임베딩 벡터의 차원 수 (50차원)
    window=3,              # 문맥 윈도우 크기: 현재 단어 기준 앞뒤 3개 단어를 문맥으로 사용
    min_count=1,           # 최소 등장 빈도: 1번 이상 나온 단어는 모두 학습 (데이터가 적으므로 1로 설정)
    workers=4,             # 학습에 사용할 CPU 스레드 수 (병렬 처리로 속도 향상)
)

print("학습된 단어 목록:", model.wv.index_to_key)  # 어휘 사전에 등록된 모든 단어 출력

target_word = "실적"              # 유사 단어를 검색할 기준 단어
if target_word in model.wv:       # 해당 단어가 어휘 사전에 존재하는지 확인
    print(f"'{target_word}' 벡터:", model.wv[target_word])  # 해당 단어의 50차원 임베딩 벡터 출력
    # 코사인 유사도 기준으로 가장 유사한 단어 Top-10과 유사도 점수 출력
    print(f"'{target_word}'와 유사한 단어:", model.wv.most_similar(target_word))
else:
    print(f"'{target_word}' 단어가 포함되지 않았습니다.")  # 단어가 없을 경우 안내 메시지 출력
