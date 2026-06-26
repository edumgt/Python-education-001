import numpy as np
import pandas as pd
from datetime import date
import warnings
warnings.filterwarnings("ignore")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _compute_rsi(prices, n=14):
    s = pd.Series(prices.astype(float))
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, 1e-8)
    return (100 - 100 / (1 + rs)).values


def _compute_macd(prices, fast=12, slow=26, signal=9):
    s = pd.Series(prices.astype(float))
    macd_line = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line.values, signal_line.values


def _to_float_list(arr):
    return [float(x) if not (np.isnan(x) or np.isinf(x)) else None for x in arr]


# ── 1. Data loader ─────────────────────────────────────────────────────────────

def get_stock_data(ticker: str, start: str = "2020-01-01", end: str = None):
    if end is None:
        end = date.today().isoformat()
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if len(df) > 50:
            close = df["Close"].squeeze().dropna()
            prices = close.values.flatten().astype(np.float32)
            dates = [d.strftime("%Y-%m-%d") for d in close.index]
            return prices, dates, True
    except Exception:
        pass

    # Synthetic fallback
    np.random.seed(42)
    n = 500
    t = np.arange(n, dtype=float)
    prices = (50000 + 80 * t + 4000 * np.sin(t / 40) + np.random.normal(0, 600, n)).astype(np.float32)
    bdays = pd.bdate_range(start="2020-01-02", periods=n)
    dates = [d.strftime("%Y-%m-%d") for d in bdays]
    return prices, dates, False


# ── 2. Technical Analysis ──────────────────────────────────────────────────────

def technical_analysis(prices: np.ndarray, dates: list):
    s = pd.Series(prices.astype(float))

    sma5 = s.rolling(5).mean()
    sma20 = s.rolling(20).mean()
    sma60 = s.rolling(60).mean()
    std20 = s.rolling(20).std()
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    rsi = pd.Series(_compute_rsi(prices))
    macd_vals, signal_vals = _compute_macd(prices)

    cur_rsi = float(rsi.dropna().iloc[-1]) if len(rsi.dropna()) else 50.0
    cur_macd = float(macd_vals[~np.isnan(macd_vals)][-1]) if any(~np.isnan(macd_vals)) else 0.0

    recent = 252
    slice_prices = prices[-recent:].tolist()
    slice_dates = dates[-recent:]

    return {
        "prices": slice_prices,
        "dates": slice_dates,
        "sma5": _to_float_list(sma5.values[-recent:]),
        "sma20": _to_float_list(sma20.values[-recent:]),
        "sma60": _to_float_list(sma60.values[-recent:]),
        "upper_bb": _to_float_list(upper_bb.values[-recent:]),
        "lower_bb": _to_float_list(lower_bb.values[-recent:]),
        "rsi": _to_float_list(rsi.values[-recent:]),
        "macd": _to_float_list(macd_vals[-recent:]),
        "signal_line": _to_float_list(signal_vals[-recent:]),
        "current_rsi": cur_rsi,
        "current_macd": cur_macd,
        "current_price": float(prices[-1]),
        "price_change": float((prices[-1] - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0.0,
    }


# ── 3. ARIMA Forecast ─────────────────────────────────────────────────────────

def arima_forecast(prices: np.ndarray, dates: list, forecast_days: int = 22):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    adf_p = adfuller(prices)[1]
    d = 1 if adf_p > 0.05 else 0

    model = ARIMA(prices, order=(2, d, 2))
    fitted = model.fit()

    fc = fitted.get_forecast(steps=forecast_days)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    mean_vals = np.asarray(mean_fc) if not hasattr(mean_fc, "values") else mean_fc.values
    lower = ci.iloc[:, 0].values if hasattr(ci, "iloc") else ci[:, 0]
    upper = ci.iloc[:, 1].values if hasattr(ci, "iloc") else ci[:, 1]

    last_date = pd.Timestamp(dates[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    future_strs = [d.strftime("%Y-%m-%d") for d in future_dates]

    # Recent 6 months of history for chart
    cutoff = pd.Timestamp(dates[-1]) - pd.DateOffset(months=6)
    recent_mask = [pd.Timestamp(d) >= cutoff for d in dates]
    recent_prices = [float(p) for p, m in zip(prices, recent_mask) if m]
    recent_dates = [d for d, m in zip(dates, recent_mask) if m]

    return {
        "recent_prices": recent_prices,
        "recent_dates": recent_dates,
        "forecast": mean_vals.tolist(),
        "lower_ci": lower.tolist(),
        "upper_ci": upper.tolist(),
        "forecast_dates": future_strs,
        "current_price": float(prices[-1]),
        "predicted_price": float(mean_vals[-1]),
        "change_pct": float((mean_vals[-1] - prices[-1]) / prices[-1] * 100),
        "aic": round(float(fitted.aic), 2),
        "bic": round(float(fitted.bic), 2),
        "order": f"ARIMA(2,{d},2)",
    }


# ── 4. LSTM Prediction ────────────────────────────────────────────────────────

def lstm_predict(prices: np.ndarray, seq_len: int = 20, epochs: int = 80):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise RuntimeError("PyTorch is required for LSTM prediction")

    torch.manual_seed(42)
    np.random.seed(42)

    p_min, p_max = prices.min(), prices.max()
    norm = (prices - p_min) / (p_max - p_min + 1e-8)

    X_list, y_list = [], []
    for i in range(len(norm) - seq_len):
        X_list.append(norm[i: i + seq_len])
        y_list.append(norm[i + seq_len])

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)

    split = int(len(X_all) * 0.8)
    X_train = torch.tensor(X_all[:split]).unsqueeze(-1)
    X_test = torch.tensor(X_all[split:]).unsqueeze(-1)
    y_train = torch.tensor(y_all[:split]).unsqueeze(-1)
    y_test = torch.tensor(y_all[split:]).unsqueeze(-1)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.2)
            self.drop = nn.Dropout(0.2)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    loss_history = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_train))
        ep_loss = 0.0
        for start in range(0, len(X_train), 32):
            idx = perm[start: start + 32]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        scheduler.step()
        loss_history.append(ep_loss / len(X_train))

    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test).numpy().flatten()
        true_norm = y_test.numpy().flatten()

    pred_real = pred_norm * (p_max - p_min) + p_min
    true_real = true_norm * (p_max - p_min) + p_min
    mae = float(np.mean(np.abs(pred_real - true_real)))
    rmse = float(np.sqrt(np.mean((pred_real - true_real) ** 2)))

    last_seq = torch.tensor(norm[-seq_len:]).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        next_norm = model(last_seq).item()
    next_price = float(next_norm * (p_max - p_min) + p_min)

    return {
        "actual": true_real.tolist(),
        "predicted": pred_real.tolist(),
        "loss_history": loss_history,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "current_price": float(prices[-1]),
        "next_price": round(next_price, 2),
        "change_pct": round((next_price - prices[-1]) / prices[-1] * 100, 2),
        "seq_len": seq_len,
        "epochs": epochs,
    }


# ── 5. Logistic Trade Signal ──────────────────────────────────────────────────

def logistic_trade_signal(prices: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    rsi = _compute_rsi(prices)
    macd_vals, _ = _compute_macd(prices)
    returns = np.diff(prices.astype(float)) / prices[:-1]

    rsi_f = rsi[:-1]
    macd_f = macd_vals[:-1]
    y = (returns > 0).astype(int)
    valid = ~(np.isnan(rsi_f) | np.isnan(macd_f))
    X = np.column_stack([rsi_f[valid], macd_f[valid]])
    y = y[valid]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    split = int(len(X_s) * 0.8)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_s[:split], y[:split])

    y_pred = clf.predict(X_s[split:])
    y_prob = clf.predict_proba(X_s[split:])[:, 1]
    acc = accuracy_score(y[split:], y_pred)

    cur_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
    cur_macd = float(macd_vals[-1]) if not np.isnan(macd_vals[-1]) else 0.0
    if not (np.isnan(rsi[-1]) or np.isnan(macd_vals[-1])):
        cur_feat = scaler.transform([[cur_rsi, cur_macd]])
        cur_sig = int(clf.predict(cur_feat)[0])
        cur_prob = float(clf.predict_proba(cur_feat)[0, 1])
    else:
        cur_sig, cur_prob = 0, 0.5

    return {
        "accuracy": round(float(acc), 4),
        "signal": "BUY" if cur_sig == 1 else "SELL",
        "buy_probability": round(cur_prob, 4),
        "current_rsi": round(cur_rsi, 2),
        "current_macd": round(cur_macd, 4),
        "test_signals": y_pred.tolist(),
        "test_probs": y_prob.tolist(),
        "test_actuals": y[split:].tolist(),
        "current_price": float(prices[-1]),
    }


# ── 6. SVM Market Phase ───────────────────────────────────────────────────────

def svm_market_phase(prices: np.ndarray):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    s = pd.Series(prices.astype(float))
    rsi = _compute_rsi(prices)
    macd_vals, _ = _compute_macd(prices)
    vol = s.pct_change().rolling(20).std().values
    returns = np.diff(prices.astype(float)) / prices[:-1]

    y = (returns > 0).astype(int)
    rsi_f, macd_f, vol_f = rsi[:-1], macd_vals[:-1], vol[:-1]
    valid = ~(np.isnan(rsi_f) | np.isnan(macd_f) | np.isnan(vol_f))
    X = np.column_stack([rsi_f[valid], macd_f[valid], vol_f[valid]])
    y = y[valid]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    split = int(len(X_s) * 0.8)
    clf = SVC(kernel="rbf", probability=True)
    clf.fit(X_s[:split], y[:split])

    y_pred = clf.predict(X_s[split:])
    y_prob = clf.predict_proba(X_s[split:])[:, 1]
    acc = accuracy_score(y[split:], y_pred)

    cur_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
    cur_macd = float(macd_vals[-1]) if not np.isnan(macd_vals[-1]) else 0.0
    cur_vol = float(vol[-1]) if not np.isnan(vol[-1]) else 0.02
    cur_feat = scaler.transform([[cur_rsi, cur_macd, cur_vol]])
    cur_phase = int(clf.predict(cur_feat)[0])
    cur_conf = float(clf.predict_proba(cur_feat)[0, 1])

    return {
        "accuracy": round(float(acc), 4),
        "market_phase": "BULL" if cur_phase == 1 else "BEAR",
        "confidence": round(cur_conf, 4),
        "current_rsi": round(cur_rsi, 2),
        "current_macd": round(cur_macd, 4),
        "test_phases": y_pred.tolist(),
        "test_probs": y_prob.tolist(),
        "test_actuals": y[split:].tolist(),
        "current_price": float(prices[-1]),
    }


# ── 7. K-Means Cluster ────────────────────────────────────────────────────────

def kmeans_cluster(tickers: list):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    records = []
    for t in tickers:
        try:
            prices, _, _ = get_stock_data(t, start="2022-01-01")
            if len(prices) < 50:
                continue
            rets = np.diff(prices.astype(float)) / prices[:-1]
            records.append({
                "ticker": t,
                "avg_return": float(np.mean(rets) * 252),
                "volatility": float(np.std(rets) * np.sqrt(252)),
            })
        except Exception:
            continue

    if len(records) < 3:
        return {"error": "Need at least 3 valid tickers"}

    df = pd.DataFrame(records)
    X = df[["avg_return", "volatility"]].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    k = min(3, len(records))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_s)
    cluster_names = ["방어주", "가치주", "성장주"]

    result = []
    for i, rec in enumerate(records):
        rec["cluster"] = int(labels[i])
        rec["cluster_name"] = cluster_names[labels[i] % len(cluster_names)]
        result.append(rec)

    return {"clusters": result, "k": k}


# ── 8. CNN+LSTM Hybrid ────────────────────────────────────────────────────────

def cnn_lstm_hybrid(prices: np.ndarray, seq_len: int = 30, epochs: int = 60):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise RuntimeError("PyTorch is required")

    torch.manual_seed(42)
    np.random.seed(42)

    # Binary classification: next day up/down
    returns = np.diff(prices.astype(float)) / prices[:-1]
    labels = (returns > 0).astype(np.float32)

    # Normalize prices
    p_min, p_max = prices.min(), prices.max()
    norm = (prices - p_min) / (p_max - p_min + 1e-8)

    X_list, y_list = [], []
    for i in range(len(norm) - seq_len):
        X_list.append(norm[i: i + seq_len])
        y_list.append(labels[i + seq_len - 1])

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)

    split = int(len(X_all) * 0.8)
    X_train = torch.tensor(X_all[:split]).unsqueeze(1)  # (N, 1, T) for Conv1d
    X_test = torch.tensor(X_all[split:]).unsqueeze(1)
    y_train = torch.tensor(y_all[:split]).unsqueeze(-1)
    y_test = torch.tensor(y_all[split:]).unsqueeze(-1)

    class CnnLstm(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.lstm = nn.LSTM(16, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            x = self.cnn(x).transpose(1, 2)  # (B, T/2, 16)
            out, _ = self.lstm(x)
            return self.sig(self.fc(out[:, -1, :]))

    model = CnnLstm()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_train))
        ep_loss = 0.0
        for s in range(0, len(X_train), 32):
            idx = perm[s: s + 32]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        loss_history.append(ep_loss / len(X_train))

    model.eval()
    with torch.no_grad():
        prob = model(X_test).numpy().flatten()
        true = y_test.numpy().flatten()

    pred = (prob > 0.5).astype(int)
    acc = float((pred == true.astype(int)).mean())

    last_seq = torch.tensor(norm[-seq_len:]).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        cur_prob = float(model(last_seq).item())

    return {
        "accuracy": round(acc, 4),
        "loss_history": loss_history,
        "test_probs": prob.tolist(),
        "test_actuals": true.tolist(),
        "signal": "BUY" if cur_prob > 0.5 else "SELL",
        "buy_probability": round(cur_prob, 4),
        "current_price": float(prices[-1]),
    }
