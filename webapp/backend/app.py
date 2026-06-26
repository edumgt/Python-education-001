import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS


def _convert(obj):
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_convert(v) for v in obj.tolist()]
    return obj


class NumpyJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kw):
        return super().dumps(_convert(obj), **kw)

    def response(self, *args, **kw):
        args = tuple(_convert(a) for a in args)
        return super().response(*args, **kw)

from stock_models import (
    get_stock_data,
    technical_analysis,
    arima_forecast,
    lstm_predict,
    logistic_trade_signal,
    svm_market_phase,
    kmeans_cluster,
    cnn_lstm_hybrid,
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
CORS(app)


# ── Static files ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ── API helpers ────────────────────────────────────────────────────────────────

def _ok(data):
    return jsonify({"success": True, "data": data})


def _err(msg, code=400):
    return jsonify({"success": False, "error": msg}), code


# ── Model catalogue ────────────────────────────────────────────────────────────

MODELS = [
    {
        "id": "technical",
        "name": "기술적 분석",
        "name_en": "Technical Analysis",
        "description": "이동평균(SMA), 볼린저밴드, RSI, MACD 시각화",
        "category": "analysis",
        "icon": "chart-line",
        "fast": True,
    },
    {
        "id": "arima",
        "name": "ARIMA 예측",
        "name_en": "ARIMA Forecast",
        "description": "통계 기반 시계열 모델로 1개월 주가 예측",
        "category": "forecast",
        "icon": "wave-square",
        "fast": True,
    },
    {
        "id": "lstm",
        "name": "LSTM 예측",
        "name_en": "LSTM Prediction",
        "description": "PyTorch LSTM 딥러닝으로 내일 주가 예측",
        "category": "forecast",
        "icon": "brain",
        "fast": False,
    },
    {
        "id": "logistic",
        "name": "로지스틱 매매신호",
        "name_en": "Logistic Trade Signal",
        "description": "RSI/MACD 기반 매수/매도 신호 분류",
        "category": "signal",
        "icon": "signal",
        "fast": True,
    },
    {
        "id": "svm",
        "name": "SVM 시장국면",
        "name_en": "SVM Market Phase",
        "description": "RBF SVM으로 현재 시장 상승/하락 국면 판단",
        "category": "signal",
        "icon": "expand-arrows-alt",
        "fast": True,
    },
    {
        "id": "cnn_lstm",
        "name": "CNN+LSTM 하이브리드",
        "name_en": "CNN+LSTM Hybrid",
        "description": "CNN 패턴 추출 → LSTM 시간 학습 상승/하락 분류",
        "category": "signal",
        "icon": "project-diagram",
        "fast": False,
    },
    {
        "id": "kmeans",
        "name": "K-Means 군집화",
        "name_en": "K-Means Clustering",
        "description": "수익률·변동성 기반 종목 자동 군집 분류",
        "category": "cluster",
        "icon": "object-group",
        "fast": True,
    },
]


@app.route("/api/models")
def get_models():
    return _ok(MODELS)


# ── Stock data endpoint ────────────────────────────────────────────────────────

@app.route("/api/stock-data", methods=["POST"])
def stock_data_endpoint():
    body = request.get_json(silent=True) or {}
    ticker = body.get("ticker", "078935.KS").strip()
    start = body.get("start", "2020-01-01")
    end = body.get("end", None)

    try:
        prices, dates, real = get_stock_data(ticker, start=start, end=end)
        return _ok({
            "ticker": ticker,
            "prices": prices.tolist(),
            "dates": dates,
            "is_real": real,
            "count": len(prices),
            "current_price": float(prices[-1]),
            "price_change": float((prices[-1] - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0.0,
        })
    except Exception as e:
        return _err(str(e))


# ── Unified predict endpoint ───────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    ticker = body.get("ticker", "078935.KS").strip()
    model_id = body.get("model", "technical")
    params = body.get("params", {})

    try:
        prices, dates, is_real = get_stock_data(ticker)
    except Exception as e:
        return _err(f"데이터 로드 실패: {e}")

    t0 = time.time()
    try:
        if model_id == "technical":
            result = technical_analysis(prices, dates)

        elif model_id == "arima":
            forecast_days = int(params.get("forecast_days", 22))
            result = arima_forecast(prices, dates, forecast_days=forecast_days)

        elif model_id == "lstm":
            seq_len = int(params.get("seq_len", 20))
            epochs = int(params.get("epochs", 80))
            result = lstm_predict(prices, seq_len=seq_len, epochs=epochs)

        elif model_id == "logistic":
            result = logistic_trade_signal(prices)

        elif model_id == "svm":
            result = svm_market_phase(prices)

        elif model_id == "cnn_lstm":
            seq_len = int(params.get("seq_len", 30))
            epochs = int(params.get("epochs", 60))
            result = cnn_lstm_hybrid(prices, seq_len=seq_len, epochs=epochs)

        elif model_id == "kmeans":
            tickers = params.get("tickers", [ticker, "005930.KS", "035420.KS", "000660.KS", "035720.KS"])
            result = kmeans_cluster(tickers)

        else:
            return _err(f"알 수 없는 모델: {model_id}")

    except Exception as e:
        traceback.print_exc()
        return _err(f"모델 실행 오류: {e}", 500)

    elapsed = round(time.time() - t0, 2)
    return _ok({
        "model": model_id,
        "ticker": ticker,
        "is_real": is_real,
        "elapsed_sec": elapsed,
        **result,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Stock Prediction API — http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
