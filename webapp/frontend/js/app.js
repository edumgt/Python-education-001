/* ── State ── */
const state = {
  activeModel: 'technical',
  ticker: '078935.KS',
  charts: {},
  lastResult: null,
};

const API = window.location.origin;

/* ── Utils ── */
const $ = id => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; };
const fmt = n => n == null ? '—' : (typeof n === 'number' ? n.toLocaleString('ko-KR', { maximumFractionDigits: 2 }) : n);
const fmtPct = n => n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
const fmtDir = n => n > 0 ? 'up' : n < 0 ? 'down' : 'neutral';

function toast(msg, type = 'info') {
  const wrap = $('toastWrap');
  const t = el('div', 'toast');
  t.style.background = type === 'error' ? '#dc2626' : type === 'success' ? '#059669' : '#1c1c1e';
  t.textContent = msg;
  wrap.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function destroyCharts() {
  Object.values(state.charts).forEach(c => { try { c.destroy(); } catch (_) {} });
  state.charts = {};
}

function setStatus(type, msg) {
  const bar = $('statusBar');
  bar.className = `status-bar visible ${type}`;
  bar.innerHTML = type === 'loading'
    ? `<div class="spinner"></div><span>${msg}</span>`
    : `<i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i><span>${msg}</span>`;
}

function clearStatus() {
  $('statusBar').className = 'status-bar';
}

/* ── Model sidebar ── */
async function loadModels() {
  try {
    const res = await fetch(`${API}/api/models`);
    const json = await res.json();
    renderSidebar(json.data);
  } catch (e) {
    renderSidebar(FALLBACK_MODELS);
  }
}

const FALLBACK_MODELS = [
  { id: 'technical', name: '기술적 분석', name_en: 'Technical Analysis', icon: 'chart-line', fast: true, category: 'analysis', description: '이동평균, 볼린저밴드, RSI, MACD' },
  { id: 'arima',     name: 'ARIMA 예측',  name_en: 'ARIMA Forecast',    icon: 'wave-square', fast: true, category: 'forecast', description: '통계 기반 1개월 주가 예측' },
  { id: 'lstm',      name: 'LSTM 예측',   name_en: 'LSTM Prediction',   icon: 'brain', fast: false, category: 'forecast', description: 'PyTorch LSTM 딥러닝 예측' },
  { id: 'logistic',  name: '로지스틱 매매신호', name_en: 'Trade Signal', icon: 'signal', fast: true,  category: 'signal',   description: 'RSI/MACD 기반 매수/매도' },
  { id: 'svm',       name: 'SVM 시장국면',name_en: 'Market Phase',      icon: 'expand-arrows-alt', fast: true, category: 'signal', description: 'RBF SVM 상승/하락 국면' },
  { id: 'cnn_lstm',  name: 'CNN+LSTM',   name_en: 'CNN+LSTM Hybrid',    icon: 'project-diagram', fast: false, category: 'signal', description: 'CNN 패턴 + LSTM 시간 학습' },
  { id: 'kmeans',    name: 'K-Means 군집', name_en: 'Clustering',       icon: 'object-group', fast: true,  category: 'cluster', description: '종목 자동 군집 분류' },
];

const CATEGORY_LABELS = {
  analysis: '분석',
  forecast: '예측',
  signal:   '매매 신호',
  cluster:  '군집화',
};

function renderSidebar(models) {
  const container = $('modelList');
  container.innerHTML = '';

  const cats = {};
  models.forEach(m => { (cats[m.category] = cats[m.category] || []).push(m); });

  Object.entries(cats).forEach(([cat, items]) => {
    const label = el('div', 'sidebar-label', CATEGORY_LABELS[cat] || cat);
    container.appendChild(label);

    items.forEach(m => {
      const btn = el('button', `model-btn${m.id === state.activeModel ? ' active' : ''}`);
      btn.dataset.id = m.id;
      btn.innerHTML = `
        <div class="icon"><i class="fas fa-${m.icon}"></i></div>
        <div class="model-btn-text">
          <div class="model-btn-name">${m.name}</div>
          <div class="model-btn-desc">${m.description}</div>
        </div>
        <span class="model-badge ${m.fast ? 'fast' : ''}">${m.fast ? '빠름' : 'GPU'}</span>
      `;
      btn.addEventListener('click', () => selectModel(m.id));
      container.appendChild(btn);
    });

    container.appendChild(el('div', 'sidebar-divider'));
  });
}

function selectModel(id) {
  state.activeModel = id;
  document.querySelectorAll('.model-btn').forEach(b => b.classList.toggle('active', b.dataset.id === id));
  updateParamsPanel(id);
  showEmptyState();
  clearStatus();
}

/* ── Params panel ── */
const PARAM_DEFS = {
  arima:    [{ key: 'forecast_days', label: '예측 기간 (거래일)', default: 22, min: 5, max: 60 }],
  lstm:     [{ key: 'seq_len', label: '시퀀스 길이 (일)', default: 20, min: 5, max: 60 }, { key: 'epochs', label: '학습 에폭', default: 80, min: 20, max: 200 }],
  cnn_lstm: [{ key: 'seq_len', label: '시퀀스 길이 (일)', default: 30, min: 10, max: 60 }, { key: 'epochs', label: '학습 에폭', default: 60, min: 20, max: 150 }],
  kmeans:   [],
};

function updateParamsPanel(modelId) {
  const panel = $('paramsPanel');
  const defs = PARAM_DEFS[modelId];
  if (!defs || defs.length === 0) { panel.className = 'params-panel'; return; }

  panel.innerHTML = defs.map(d =>
    `<div class="input-group">
      <label>${d.label}</label>
      <input type="number" id="param_${d.key}" value="${d.default}" min="${d.min}" max="${d.max}">
    </div>`
  ).join('');
  panel.className = 'params-panel visible';
}

function getParams() {
  const defs = PARAM_DEFS[state.activeModel] || [];
  const p = {};
  defs.forEach(d => {
    const el = $(`param_${d.key}`);
    if (el) p[d.key] = Number(el.value);
  });
  if (state.activeModel === 'kmeans') {
    p.tickers = getTickerChips();
  }
  return p;
}

function getTickerChips() {
  return Array.from(document.querySelectorAll('.ticker-chip.active')).map(c => c.dataset.ticker);
}

/* ── Ticker chips ── */
const QUICK_TICKERS = [
  { t: '078935.KS', label: 'GS피앤엘' },
  { t: '005930.KS', label: '삼성전자' },
  { t: '035420.KS', label: 'NAVER' },
  { t: '000660.KS', label: 'SK하이닉스' },
  { t: '035720.KS', label: 'Kakao' },
  { t: 'AAPL',       label: 'Apple' },
  { t: 'NVDA',       label: 'NVIDIA' },
  { t: 'TSLA',       label: 'Tesla' },
];

function renderTickerChips() {
  const wrap = $('tickerChips');
  QUICK_TICKERS.forEach(({ t, label }) => {
    const chip = el('span', `ticker-chip${t === state.ticker ? ' active' : ''}`);
    chip.dataset.ticker = t;
    chip.textContent = label;
    chip.addEventListener('click', () => {
      state.ticker = t;
      $('tickerInput').value = t;
      document.querySelectorAll('.ticker-chip').forEach(c =>
        c.classList.toggle('active', c.dataset.ticker === t));
    });
    wrap.appendChild(chip);
  });
}

/* ── Run prediction ── */
async function runPrediction() {
  const ticker = ($('tickerInput').value || '').trim() || state.ticker;
  state.ticker = ticker;
  document.querySelectorAll('.ticker-chip').forEach(c =>
    c.classList.toggle('active', c.dataset.ticker === ticker));

  const btn = $('runBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner" style="border-color:white;border-top-color:transparent;"></div><span>분석 중...</span>';

  setStatus('loading', `${ticker} 데이터로 ${state.activeModel.toUpperCase()} 모델 실행 중...`);
  $('resultsSection').innerHTML = '';
  destroyCharts();

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, model: state.activeModel, params: getParams() }),
    });
    const json = await res.json();

    if (!json.success) throw new Error(json.error || '서버 오류');

    state.lastResult = json.data;
    const d = json.data;
    const src = d.is_real ? `실제 데이터 (${ticker})` : `합성 데이터`;
    setStatus('success', `완료 — ${src}  |  ${d.elapsed_sec}초 소요`);
    toast(`${ticker} ${state.activeModel} 분석 완료!`, 'success');

    renderResults(state.activeModel, d);

  } catch (e) {
    setStatus('error', `오류: ${e.message}`);
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-play"></i><span>분석 실행</span>';
  }
}

/* ── Show empty state ── */
function showEmptyState() {
  destroyCharts();
  const section = $('resultsSection');
  section.innerHTML = `
    <div class="empty-state fade-in">
      <div class="empty-icon"><i class="fas fa-chart-line"></i></div>
      <div class="empty-title">분석 결과가 여기에 표시됩니다</div>
      <div class="empty-text">종목 코드를 입력하고 왼쪽 사이드바에서 모델을 선택한 뒤 <b>분석 실행</b> 버튼을 누르세요.</div>
    </div>`;
}

/* ── Render results ── */
function renderResults(modelId, d) {
  const section = $('resultsSection');
  section.innerHTML = '';

  if (modelId === 'technical') renderTechnical(section, d);
  else if (modelId === 'arima')    renderArima(section, d);
  else if (modelId === 'lstm')     renderLstm(section, d);
  else if (modelId === 'logistic') renderSignal(section, d, 'logistic');
  else if (modelId === 'svm')      renderSignal(section, d, 'svm');
  else if (modelId === 'cnn_lstm') renderSignal(section, d, 'cnn_lstm');
  else if (modelId === 'kmeans')   renderKmeans(section, d);
}

/* ── Metric card builder ── */
function metricCard(label, value, sub, cls = '') {
  return `<div class="metric-card fade-in">
    <div class="metric-label">${label}</div>
    <div class="metric-value ${cls}">${value}</div>
    ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
  </div>`;
}

function signalBadge(sig) {
  const cls = sig === 'BUY' || sig === 'BULL' ? 'buy' : 'sell';
  const icon = cls === 'buy' ? 'arrow-up' : 'arrow-down';
  const text = sig === 'BULL' ? '상승장' : sig === 'BEAR' ? '하락장' : sig;
  return `<span class="metric-badge ${cls}"><i class="fas fa-${icon}"></i>${text}</span>`;
}

/* ── Technical Analysis render ── */
function renderTechnical(section, d) {
  const pct = d.price_change;
  const metrics = el('div', 'metrics-grid fade-in');
  metrics.innerHTML =
    metricCard('현재가', `${fmt(d.current_price)}원`, '종가 기준', pct >= 0 ? 'up' : 'down') +
    metricCard('RSI (14)', `${d.current_rsi.toFixed(1)}`, d.current_rsi > 70 ? '⚠ 과매수' : d.current_rsi < 30 ? '⚠ 과매도' : '중립') +
    metricCard('MACD', `${d.current_macd.toFixed(4)}`, d.current_macd > 0 ? '▲ 상승 모멘텀' : '▼ 하락 모멘텀', d.current_macd > 0 ? 'up' : 'down') +
    metricCard('전일 대비', fmtPct(pct), '종가 기준', fmtDir(pct));
  section.appendChild(metrics);

  // Price + BB chart
  const priceCard = chartCard('가격 차트 — 이동평균 & 볼린저밴드', 'chart-line', '최근 252거래일', 'priceChart', 320);
  section.appendChild(priceCard);

  const twoCol = el('div', 'charts-grid two-col');
  twoCol.appendChild(chartCard('RSI (14일)', 'tachometer-alt', '70 = 과매수, 30 = 과매도', 'rsiChart', 200));
  twoCol.appendChild(chartCard('MACD / Signal', 'wave-square', 'EMA12 - EMA26', 'macdChart', 200));
  section.appendChild(twoCol);

  requestAnimationFrame(() => {
    state.charts.price = drawTechnicalChart('priceChart', d);
    state.charts.rsi   = drawRsiChart('rsiChart', d);
    state.charts.macd  = drawMacdChart('macdChart', d);
  });
}

/* ── ARIMA render ── */
function renderArima(section, d) {
  const metrics = el('div', 'metrics-grid fade-in');
  metrics.innerHTML =
    metricCard('현재가', `${fmt(d.current_price)}원`, '오늘 종가') +
    metricCard('1개월 예측', `${fmt(d.predicted_price)}원`, d.order, d.change_pct >= 0 ? 'up' : 'down') +
    metricCard('예상 변화율', fmtPct(d.change_pct), '현재가 대비', d.change_pct >= 0 ? 'up' : 'down') +
    metricCard('AIC', `${d.aic}`, `BIC: ${d.bic}`, 'neutral');
  section.appendChild(metrics);

  const card = chartCard(`ARIMA 예측 — ${d.order}`, 'wave-square', '최근 6개월 + 22거래일 예측', 'arimaChart', 360);
  section.appendChild(card);

  requestAnimationFrame(() => { state.charts.arima = drawArimaChart('arimaChart', d); });
}

/* ── LSTM render ── */
function renderLstm(section, d) {
  const metrics = el('div', 'metrics-grid fade-in');
  metrics.innerHTML =
    metricCard('현재가', `${fmt(d.current_price)}원`, '오늘 종가') +
    metricCard('내일 예측', `${fmt(d.next_price)}원`, `SEQ=${d.seq_len}일`, d.change_pct >= 0 ? 'up' : 'down') +
    metricCard('예상 변화율', fmtPct(d.change_pct), '현재가 대비', d.change_pct >= 0 ? 'up' : 'down') +
    metricCard('MAE', `${fmt(d.mae)}원`, `RMSE: ${fmt(d.rmse)}원`, 'neutral');
  section.appendChild(metrics);

  const twoCol = el('div', 'charts-grid two-col');
  twoCol.appendChild(chartCard('실제 vs LSTM 예측', 'brain', '테스트 세트', 'lstmChart', 300));
  twoCol.appendChild(chartCard('학습 손실 곡선 (MSE)', 'chart-area', `${d.epochs} Epochs`, 'lossChart', 300));
  section.appendChild(twoCol);

  requestAnimationFrame(() => {
    state.charts.lstm = drawLstmChart('lstmChart', d);
    state.charts.loss = drawLossChart('lossChart', d.loss_history);
  });
}

/* ── Signal render (Logistic / SVM / CNN+LSTM) ── */
function renderSignal(section, d, type) {
  const isBull = d.signal === 'BUY' || d.signal === 'BULL' || d.market_phase === 'BULL';
  const prob = d.buy_probability != null ? d.buy_probability : d.confidence;
  const accPct = (d.accuracy * 100).toFixed(1);
  const sig = d.signal || d.market_phase;
  const sigDisplay = sig === 'BULL' ? '상승장' : sig === 'BEAR' ? '하락장' : sig;

  const metrics = el('div', 'metrics-grid fade-in');
  metrics.innerHTML =
    metricCard('현재가', `${fmt(d.current_price)}원`, '오늘 기준') +
    metricCard('모델 신호', `<span class="metric-badge ${isBull ? 'buy' : 'sell'}"><i class="fas fa-${isBull ? 'arrow-up' : 'arrow-down'}"></i>${sigDisplay}</span>`, '', '') +
    metricCard('매수 확률', `${(prob * 100).toFixed(1)}%`, prob > 0.5 ? '▲ 강세' : '▼ 약세', prob > 0.5 ? 'up' : 'down') +
    metricCard('테스트 정확도', `${accPct}%`, `RSI: ${d.current_rsi}`, 'neutral');
  section.appendChild(metrics);

  const threeCol = el('div', 'charts-grid three-col');

  // Signal ring card
  const ringCard = el('div', 'chart-card fade-in');
  ringCard.innerHTML = `
    <div class="chart-card-header">
      <div class="chart-card-title"><i class="fas fa-bullseye"></i> 현재 신호</div>
    </div>
    <div class="signal-display">
      <div class="signal-ring">
        <svg width="110" height="110" viewBox="0 0 110 110">
          <circle class="signal-ring-bg" cx="55" cy="55" r="45" fill="none" stroke-width="8"/>
          <circle class="signal-ring-fill ${isBull ? 'buy' : 'sell'}" id="signalArc"
            cx="55" cy="55" r="45" fill="none" stroke-width="8"/>
        </svg>
        <div class="signal-center">
          <div class="signal-pct" id="signalPct">0%</div>
          <div class="signal-label">매수 확률</div>
        </div>
      </div>
      <div class="signal-decision ${isBull ? 'buy' : 'sell'}">${sigDisplay}</div>
      <div class="signal-meta">RSI: <b>${d.current_rsi}</b>${d.current_macd != null ? `  MACD: <b>${d.current_macd.toFixed(4)}</b>` : ''}</div>
    </div>`;
  threeCol.appendChild(ringCard);

  // Accuracy donut
  threeCol.appendChild(chartCard('예측 정확도', 'check-circle', `테스트 세트 (n=${d.test_actuals ? d.test_actuals.length : 0})`, 'accChart', 240));
  section.appendChild(threeCol);

  // Animate ring
  requestAnimationFrame(() => {
    const pct = prob;
    const circ = 2 * Math.PI * 45;
    const arc = document.getElementById('signalArc');
    const pctEl = document.getElementById('signalPct');
    if (arc) arc.style.strokeDashoffset = String(circ * (1 - pct));
    if (pctEl) {
      let cur = 0;
      const step = () => {
        cur = Math.min(cur + 2, pct * 100);
        pctEl.textContent = `${cur.toFixed(0)}%`;
        if (cur < pct * 100) requestAnimationFrame(step);
      };
      step();
    }
    if (d.test_actuals && d.test_signals) {
      state.charts.acc = drawSignalAccChart('accChart', d.test_actuals, d.test_signals);
    }
  });
}

/* ── K-Means render ── */
function renderKmeans(section, d) {
  if (d.error) {
    section.innerHTML = `<div class="empty-state"><div class="empty-icon"><i class="fas fa-exclamation-triangle"></i></div><div class="empty-title">오류</div><div class="empty-text">${d.error}</div></div>`;
    return;
  }

  const clusters = d.clusters;
  const clusterColors = ['#059669', '#2563eb', '#dc2626', '#d97706'];

  const twoCol = el('div', 'charts-grid two-col');
  twoCol.appendChild(chartCard('K-Means 군집 산점도', 'object-group', `K=${d.k} 군집`, 'clusterChart', 320));

  const tableCard = el('div', 'chart-card fade-in');
  tableCard.innerHTML = `
    <div class="chart-card-header">
      <div class="chart-card-title"><i class="fas fa-table"></i> 종목별 군집 결과</div>
    </div>
    <table class="result-table">
      <thead><tr><th>종목</th><th>군집</th><th>연간 수익률</th><th>연간 변동성</th></tr></thead>
      <tbody>
        ${clusters.map(c => `
          <tr>
            <td>${c.ticker}</td>
            <td><span class="cluster-dot" style="background:${clusterColors[c.cluster % 4]}"></span>${c.cluster_name}</td>
            <td class="${c.avg_return >= 0 ? 'metric-value up' : 'metric-value down'}" style="font-size:0.82rem">${(c.avg_return * 100).toFixed(1)}%</td>
            <td>${(c.volatility * 100).toFixed(1)}%</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
  twoCol.appendChild(tableCard);
  section.appendChild(twoCol);

  requestAnimationFrame(() => { state.charts.cluster = drawClusterChart('clusterChart', clusters); });
}

/* ── Chart card helper ── */
function chartCard(title, icon, sub, canvasId, height = 280) {
  const card = el('div', 'chart-card fade-in');
  card.innerHTML = `
    <div class="chart-card-header">
      <div class="chart-card-title"><i class="fas fa-${icon}"></i> ${title}</div>
      <div class="chart-card-sub">${sub}</div>
    </div>
    <div class="chart-wrap" style="height:${height}px">
      <canvas id="${canvasId}"></canvas>
    </div>`;
  return card;
}

/* ── Init ── */
document.addEventListener('DOMContentLoaded', () => {
  renderTickerChips();
  loadModels();
  showEmptyState();

  $('runBtn').addEventListener('click', runPrediction);
  $('tickerInput').addEventListener('keydown', e => { if (e.key === 'Enter') runPrediction(); });
  $('tickerInput').addEventListener('change', () => {
    state.ticker = $('tickerInput').value.trim();
    document.querySelectorAll('.ticker-chip').forEach(c =>
      c.classList.toggle('active', c.dataset.ticker === state.ticker));
  });

  // Default model
  selectModel('technical');
});
