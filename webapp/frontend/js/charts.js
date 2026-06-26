/* ── Chart factory ── */

const COLORS = {
  primary: '#7c3aed',
  accent:  '#2563eb',
  success: '#059669',
  danger:  '#dc2626',
  warning: '#d97706',
  gray:    '#9ca3af',
  lightPurple: 'rgba(124,58,237,0.12)',
  lightBlue:   'rgba(37,99,235,0.12)',
  lightGreen:  'rgba(5,150,105,0.12)',
  lightRed:    'rgba(220,38,38,0.12)',
};

const BASE_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { intersect: false, mode: 'index' },
  plugins: {
    legend: {
      position: 'top',
      labels: {
        usePointStyle: true,
        pointStyleWidth: 8,
        padding: 16,
        font: { size: 11, family: 'Segoe UI, system-ui, sans-serif' },
        color: '#6b7280',
      }
    },
    tooltip: {
      backgroundColor: '#1c1c1e',
      titleFont: { size: 11, weight: '600' },
      bodyFont: { size: 11 },
      padding: 10,
      cornerRadius: 8,
      caretSize: 5,
    }
  },
  scales: {
    x: {
      grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
      ticks: { color: '#9ca3af', font: { size: 10 }, maxTicksLimit: 10, maxRotation: 0 },
      border: { display: false },
    },
    y: {
      grid: { color: 'rgba(0,0,0,0.04)', drawBorder: false },
      ticks: { color: '#9ca3af', font: { size: 10 } },
      border: { display: false },
    }
  }
};

function mergeOpts(extra) {
  return deepMerge(JSON.parse(JSON.stringify(BASE_OPTS)), extra || {});
}

function deepMerge(target, source) {
  for (const k of Object.keys(source)) {
    if (source[k] && typeof source[k] === 'object' && !Array.isArray(source[k])) {
      target[k] = target[k] || {};
      deepMerge(target[k], source[k]);
    } else {
      target[k] = source[k];
    }
  }
  return target;
}

/* Downsample labels for X-axis readability */
function sparseDates(dates, maxTicks = 12) {
  if (!dates || dates.length === 0) return [];
  const step = Math.max(1, Math.floor(dates.length / maxTicks));
  return dates.map((d, i) => (i % step === 0 ? d.slice(0, 10) : ''));
}

/* ── Technical Analysis chart (price + SMA + BB) ── */
function drawTechnicalChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const labels = sparseDates(data.dates);

  const ds = [
    {
      label: '종가',
      data: data.prices,
      borderColor: COLORS.accent,
      backgroundColor: COLORS.lightBlue,
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      order: 1,
    },
    {
      label: 'SMA 5',
      data: data.sma5,
      borderColor: COLORS.warning,
      borderWidth: 1.2,
      pointRadius: 0,
      fill: false,
      borderDash: [3, 3],
      order: 2,
    },
    {
      label: 'SMA 20',
      data: data.sma20,
      borderColor: COLORS.primary,
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      order: 3,
    },
    {
      label: 'SMA 60',
      data: data.sma60,
      borderColor: COLORS.gray,
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      borderDash: [4, 4],
      order: 4,
    },
    {
      label: '볼린저 상단',
      data: data.upper_bb,
      borderColor: 'rgba(220,38,38,0.4)',
      borderWidth: 1,
      pointRadius: 0,
      fill: '+1',
      backgroundColor: 'rgba(220,38,38,0.04)',
      borderDash: [2, 2],
      order: 5,
    },
    {
      label: '볼린저 하단',
      data: data.lower_bb,
      borderColor: 'rgba(5,150,105,0.4)',
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      borderDash: [2, 2],
      order: 6,
    },
  ];

  const opts = mergeOpts({
    scales: { y: { ticks: { callback: v => v ? v.toLocaleString('ko') : '' } } }
  });

  return new Chart(ctx, { type: 'line', data: { labels, datasets: ds }, options: opts });
}

/* ── RSI chart ── */
function drawRsiChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const labels = sparseDates(data.dates);

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'RSI (14)',
        data: data.rsi,
        borderColor: COLORS.primary,
        backgroundColor: COLORS.lightPurple,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
      }]
    },
    options: mergeOpts({
      plugins: { legend: { display: false } },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: {
            callback: v => v,
            stepSize: 25,
          },
          afterDraw(chart) {
            const ctx2 = chart.ctx;
            const yAxis = chart.scales.y;
            const xAxis = chart.scales.x;
            [[70, 'rgba(220,38,38,0.2)'], [30, 'rgba(5,150,105,0.2)']].forEach(([val, color]) => {
              const y = yAxis.getPixelForValue(val);
              ctx2.save();
              ctx2.strokeStyle = color;
              ctx2.setLineDash([4, 4]);
              ctx2.lineWidth = 1;
              ctx2.beginPath();
              ctx2.moveTo(xAxis.left, y);
              ctx2.lineTo(xAxis.right, y);
              ctx2.stroke();
              ctx2.restore();
            });
          }
        }
      }
    })
  });
}

/* ── MACD chart ── */
function drawMacdChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const labels = sparseDates(data.dates);
  const hist = data.macd.map((v, i) =>
    v != null && data.signal_line[i] != null ? v - data.signal_line[i] : null
  );

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'MACD',
          data: data.macd,
          borderColor: COLORS.primary,
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          type: 'line',
          pointRadius: 0,
          order: 1,
        },
        {
          label: 'Signal',
          data: data.signal_line,
          borderColor: COLORS.danger,
          backgroundColor: 'transparent',
          borderWidth: 1.2,
          type: 'line',
          pointRadius: 0,
          borderDash: [3, 3],
          order: 2,
        },
        {
          label: 'Histogram',
          data: hist,
          backgroundColor: hist.map(v => (v == null ? 'transparent' : v >= 0 ? 'rgba(5,150,105,0.4)' : 'rgba(220,38,38,0.4)')),
          borderWidth: 0,
          order: 3,
        },
      ]
    },
    options: mergeOpts({ plugins: { legend: { labels: { filter: i => i.text !== 'Histogram' } } } })
  });
}

/* ── ARIMA forecast chart ── */
function drawArimaChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const allDates = [...data.recent_dates, ...data.forecast_dates];
  const allHistPrices = [...data.recent_prices, ...Array(data.forecast_dates.length).fill(null)];
  const allForecast   = [...Array(data.recent_dates.length).fill(null), ...data.forecast];
  const allLower      = [...Array(data.recent_dates.length).fill(null), ...data.lower_ci];
  const allUpper      = [...Array(data.recent_dates.length).fill(null), ...data.upper_ci];

  const labels = sparseDates(allDates, 14);

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: '실제 주가',
          data: allHistPrices,
          borderColor: COLORS.accent,
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          order: 1,
        },
        {
          label: 'ARIMA 예측',
          data: allForecast,
          borderColor: COLORS.danger,
          borderWidth: 2.5,
          pointRadius: 3,
          pointBackgroundColor: COLORS.danger,
          fill: false,
          order: 0,
        },
        {
          label: '95% CI 상단',
          data: allUpper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(220,38,38,0.1)',
          fill: '+1',
          pointRadius: 0,
          borderWidth: 0,
          order: 2,
        },
        {
          label: '95% CI 하단',
          data: allLower,
          borderColor: 'rgba(220,38,38,0.3)',
          borderWidth: 1,
          borderDash: [3, 3],
          fill: false,
          pointRadius: 0,
          order: 3,
        },
      ]
    },
    options: mergeOpts({
      plugins: {
        legend: {
          labels: { filter: i => !i.text.includes('CI 하단') && !i.text.includes('CI 상단') }
        }
      },
      scales: { y: { ticks: { callback: v => v ? v.toLocaleString('ko') : '' } } }
    })
  });
}

/* ── LSTM chart ── */
function drawLstmChart(canvasId, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const labels = Array.from({ length: data.actual.length }, (_, i) => String(i + 1));

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: '실제 주가',
          data: data.actual,
          borderColor: COLORS.accent,
          borderWidth: 1.8,
          pointRadius: 0,
          fill: false,
        },
        {
          label: 'LSTM 예측',
          data: data.predicted,
          borderColor: COLORS.danger,
          borderWidth: 1.8,
          borderDash: [5, 3],
          pointRadius: 0,
          fill: false,
        },
      ]
    },
    options: mergeOpts({
      scales: { y: { ticks: { callback: v => v ? v.toLocaleString('ko') : '' } } }
    })
  });
}

/* ── Loss curve chart ── */
function drawLossChart(canvasId, lossHistory) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const labels = lossHistory.map((_, i) => String(i + 1));

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: '학습 손실 (MSE)',
        data: lossHistory,
        borderColor: COLORS.primary,
        backgroundColor: COLORS.lightPurple,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
      }]
    },
    options: mergeOpts({
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Epoch', color: '#9ca3af', font: { size: 10 } } },
        y: { title: { display: true, text: 'MSE Loss', color: '#9ca3af', font: { size: 10 } } },
      }
    })
  });
}

/* ── Classification accuracy bar ── */
function drawSignalAccChart(canvasId, actuals, predictions) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const correct = predictions.filter((p, i) => p === actuals[i]).length;
  const wrong = predictions.length - correct;

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['정답', '오답'],
      datasets: [{
        data: [correct, wrong],
        backgroundColor: [COLORS.success, 'rgba(220,38,38,0.15)'],
        borderColor: ['white', 'white'],
        borderWidth: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { position: 'bottom', labels: { font: { size: 11 }, color: '#6b7280', usePointStyle: true } },
        tooltip: { backgroundColor: '#1c1c1e', cornerRadius: 8 }
      }
    }
  });
}

/* ── Scatter cluster chart ── */
function drawClusterChart(canvasId, clusters) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const colorMap = [COLORS.success, COLORS.accent, COLORS.danger, COLORS.warning];
  const grouped = {};
  clusters.forEach(c => {
    grouped[c.cluster] = grouped[c.cluster] || { label: c.cluster_name, data: [], bg: colorMap[c.cluster % 4] };
    grouped[c.cluster].data.push({ x: c.volatility * 100, y: c.avg_return * 100, ticker: c.ticker });
  });

  return new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: Object.values(grouped).map(g => ({
        label: g.label,
        data: g.data,
        backgroundColor: g.bg + 'cc',
        borderColor: g.bg,
        borderWidth: 1.5,
        pointRadius: 7,
        pointHoverRadius: 9,
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 }, color: '#6b7280', usePointStyle: true } },
        tooltip: {
          backgroundColor: '#1c1c1e',
          cornerRadius: 8,
          callbacks: {
            label: ctx => {
              const p = ctx.raw;
              return ` ${p.ticker}  수익률: ${p.y.toFixed(1)}%  변동성: ${p.x.toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: '연간 변동성 (%)', color: '#9ca3af', font: { size: 11 } },
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { color: '#9ca3af' },
        },
        y: {
          title: { display: true, text: '연간 수익률 (%)', color: '#9ca3af', font: { size: 11 } },
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { color: '#9ca3af' },
        }
      }
    }
  });
}
