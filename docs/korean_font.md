# korean_font.py 코드 설명

> 한 줄 요약: 파이썬 그래프에서 한글이 깨져 보이지 않도록 컴퓨터에 설치된 한글 폰트를 자동으로 찾아서 설정해주는 프로그램입니다.

---

## 이 코드가 하는 일

파이썬으로 그래프를 그릴 때 한글 제목이나 한글 글씨가 이상한 네모(□□□)로 깨져 보이는 경우가 있습니다.
이 코드는 마치 외국에서 한국어 문자가 잘 보이도록 "한국어 자막 설정"을 자동으로 켜주는 것과 같습니다.
컴퓨터마다 설치된 폰트가 다를 수 있어서, 여러 한글 폰트 이름을 목록에 넣어두고 그중에 실제로 설치된 것을 자동으로 골라서 사용합니다.

---

## 준비물 (import)

| 라이브러리 | 하는 일 |
|-----------|---------|
| `matplotlib` | 그래프를 그리는 도구. 여기서는 전체 설정을 바꾸기 위해 사용 |
| `matplotlib.font_manager` | 컴퓨터에 설치된 폰트 목록을 관리하고 검색하는 도구 |

---

## 코드 흐름 (단계별 설명)

### 1단계: 사용할 한글 폰트 후보 목록 만들기
```python
_CANDIDATES = [
    'NanumGothic', 'Malgun Gothic', 'AppleGothic',
    'Noto Sans CJK KR', 'NanumBarunGothic', 'NanumSquareRound',
]
```
> 📌 **쉬운 설명:** 한글을 보여줄 수 있는 폰트 이름들을 목록으로 만듭니다. `NanumGothic(나눔고딕)`은 리눅스에, `Malgun Gothic(맑은 고딕)`은 윈도우에, `AppleGothic`은 맥에 기본으로 설치되어 있습니다. 여러 운영 체제에서 다 잘 동작하도록 여러 폰트를 준비해둔 것입니다.

---

### 2단계: 지금 컴퓨터에 설치된 폰트 목록 가져오기
```python
_available = {f.name for f in font_manager.fontManager.ttflist}
```
> 📌 **쉬운 설명:** 현재 컴퓨터에 실제로 설치된 모든 폰트의 이름을 집합(set)으로 모읍니다. `font_manager.fontManager.ttflist`는 "내 컴퓨터에 있는 폰트 명단"이고, 여기서 이름만 꺼내서 `_available`에 담습니다. 마치 냉장고 문을 열어서 어떤 식재료가 있는지 목록을 확인하는 것과 같습니다.

---

### 3단계: 후보 중에서 설치된 폰트 하나 고르기
```python
_font = next((f for f in _CANDIDATES if f in _available), None)
```
> 📌 **쉬운 설명:** 후보 목록을 앞에서부터 하나씩 확인해서 컴퓨터에 실제로 설치된 첫 번째 폰트를 고릅니다. 만약 하나도 설치되어 있지 않으면 `None`(아무것도 없음)이 됩니다. 마치 편의점에서 "삼각김밥 중에 참치, 불고기, 연어 순서로 있으면 첫 번째 것 주세요"라고 하는 것과 같습니다.

---

### 4단계: 찾은 폰트를 그래프 전체 기본 설정으로 적용하기
```python
if _font:
    mpl.rcParams['font.family'] = _font
mpl.rcParams['axes.unicode_minus'] = False
```
> 📌 **쉬운 설명:** 한글 폰트를 찾은 경우에만 matplotlib의 기본 글꼴을 그 폰트로 바꿉니다. `rcParams`는 matplotlib의 모든 설정을 저장해 두는 큰 설정함입니다. 두 번째 줄 `axes.unicode_minus = False`는 마이너스 기호(-)가 이상하게 깨져 보이는 문제를 막아주는 설정입니다.

---

## 핵심 개념 한눈에 보기

| 용어 | 쉬운 설명 |
|------|---------|
| **폰트(Font)** | 글자를 어떤 모양으로 보여줄지 정의한 파일. 폰트마다 글씨체가 다름 |
| **한글 폰트** | 한글을 표시할 수 있는 폰트. 영어 전용 폰트에는 한글 글자 모양이 없어서 깨짐 |
| **matplotlib** | 파이썬에서 그래프를 그리는 가장 유명한 도구 |
| **rcParams** | matplotlib의 전역 설정을 담은 딕셔너리(설정함). 이걸 바꾸면 이후 모든 그래프에 적용됨 |
| **font_manager** | 컴퓨터에 설치된 폰트들을 찾고 관리하는 matplotlib의 도구 |
| **ttflist** | 컴퓨터에서 찾은 모든 TTF(트루타입) 폰트 파일 목록 |
| **TTF(TrueType Font)** | 가장 널리 쓰이는 폰트 파일 형식 (.ttf 파일) |
| `{f.name for f in ...}` | 목록의 각 항목에서 이름만 꺼내 집합(set)으로 만드는 파이썬 문법 (집합 내포) |
| `next(..., None)` | 조건에 맞는 첫 번째 값을 꺼내는 함수. 없으면 None 반환 |
| **NanumGothic** | 네이버가 만든 무료 한글 폰트. 리눅스/맥에서 많이 사용 |
| **Malgun Gothic** | 맑은 고딕. 윈도우에 기본으로 설치된 한글 폰트 |
| **AppleGothic** | 애플(맥/아이폰)에 기본으로 설치된 한글 폰트 |
| **unicode_minus** | matplotlib이 마이너스 기호(-)를 그릴 때 유니코드 문자를 쓸지 말지 결정하는 설정 |
| **`# noqa: F401`** | 다른 파일에서 `import korean_font`만 해도 이 코드가 자동 실행되는데, 직접 사용하지 않아도 된다고 코드 검사 도구에게 알려주는 주석 |

---

## 실행 결과

이 파일은 직접 그래프나 파일을 만들지 않습니다. 대신 다른 파일에서 `import korean_font`를 하는 순간 자동으로 실행되어 matplotlib의 한글 폰트 설정을 바꿔줍니다.

예를 들어 `TransformerAttention.py`와 `YfinanceNormalize.py` 파일 맨 위에 다음 줄이 있습니다:

```python
import korean_font  # noqa: F401
```

이 한 줄 덕분에 그 파일의 모든 그래프에서 한글 제목과 레이블이 깨지지 않고 깔끔하게 출력됩니다.
