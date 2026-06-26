#!/bin/bash
# Stock ML Web App 실행 스크립트
# 사용법: bash webapp/start.sh

set -e
cd "$(dirname "$0")/.."

echo "======================================"
echo "  Stock ML — 주식 예측 AI 대시보드"
echo "======================================"

# 가상환경 활성화
if [ -d "venv" ]; then
  source venv/bin/activate
  echo "  ✓ 가상환경 활성화"
else
  echo "  ✗ venv 폴더 없음 — python3 -m venv venv 후 pip install -r requirements.txt 실행"
  exit 1
fi

# 필수 패키지 확인
python -c "import flask, flask_cors, statsmodels, torch, sklearn" 2>/dev/null || {
  echo "  패키지 설치 중..."
  pip install flask flask-cors statsmodels torch --index-url https://download.pytorch.org/whl/cpu -q
}

PORT=${PORT:-5000}
echo "  → http://localhost:${PORT} 에서 접속하세요"
echo "======================================"
echo ""

PORT=$PORT python webapp/backend/app.py
