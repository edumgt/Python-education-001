import os
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 65)
print("  Transformer Attention: 주가 방향 분류 실습")
print("=" * 65)
print()
print("  Attention 매커니즘 비유:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  📖 책을 읽을 때 '중요한 단어'에 밑줄 긋는 것처럼   │")
print("  │     Attention은 시퀀스에서 어느 날짜가 예측에        │")
print("  │     중요한지 가중치로 학습합니다                      │")
print("  │                                                       │")
print("  │  Q(Query)  : 지금 예측하려는 것                      │")
print("  │  K(Key)    : 과거 각 날의 특성 요약                  │")
print("  │  V(Value)  : 실제로 전달할 정보                      │")
print("  │  Score = softmax(Q·Kᵀ / √d) × V                     │")
print("  │                                                       │")
print("  │  Multi-Head: 여러 관점에서 동시에 주의를 기울임      │")
print("  └──────────────────────────────────────────────────────┘")

torch.manual_seed(42)
np.random.seed(42)

# ── 1. 데이터 생성 ─────────────────────────────────────────
print("\n[1/8] 주가 시계열 로드 중 (078935.KS - GS피앤엘)...")
time.sleep(0.5)

TICKER = '078935.KS'
prices = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    days = 500
    t = np.arange(days, dtype=float)
    prices = (100 + 0.1 * t + 8 * np.sin(t / 25) + 3 * np.sin(t / 7) + np.random.normal(0, 1.8, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
returns = np.diff(prices) / prices[:-1]
print(f"   → {days}일치 주가  |  수익률 범위: {returns.min():.3f} ~ {returns.max():.3f}")
time.sleep(0.3)

# ── 2. 슬라이딩 윈도우 데이터셋 ───────────────────────────
print("\n[2/8] 슬라이딩 윈도우 데이터셋 구성 중...")
print("   입력: 과거 30일 수익률  |  출력: 다음 날 상승(1)/하락(0)")
time.sleep(0.5)
SEQ_LEN = 30
D_MODEL = 16   # Transformer 임베딩 차원 (짝수여야 함)

X_list, y_list = [], []
for i in range(len(returns) - SEQ_LEN):
    X_list.append(returns[i:i + SEQ_LEN])
    y_list.append(1 if returns[i + SEQ_LEN] > 0 else 0)

X_np = np.array(X_list, dtype=np.float32)    # (N, 30)
y_np = np.array(y_list, dtype=np.int64)

X_tensor = torch.tensor(X_np).unsqueeze(-1)  # (N, 30, 1)
y_tensor = torch.tensor(y_np)

split = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]
print(f"   → 학습: {X_train.shape}  테스트: {X_test.shape}")
time.sleep(0.3)

# ── 3. Positional Encoding ─────────────────────────────────
print("\n[3/8] Positional Encoding 준비 중...")
print("   Transformer는 순서를 모르므로 위치 정보를 sin/cos로 주입합니다")
time.sleep(0.5)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── 4. Transformer 분류 모델 ───────────────────────────────
print("\n[4/8] Transformer 분류 모델 정의 중...")
print("   Linear 입력 투영 → Positional Encoding →")
print("   TransformerEncoder(Multi-Head Self-Attention + FFN) →")
print("   Global Average Pooling → Linear 분류기")
time.sleep(0.6)


class StockTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=16, nhead=4, num_layers=2,
                 dim_ff=64, dropout=0.2, seq_len=30):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x: (B, seq_len, 1)
        x = self.input_proj(x)          # (B, seq_len, d_model)
        x = self.pos_enc(x)             # 위치 정보 추가
        x = self.encoder(x)             # Self-Attention (B, seq_len, d_model)
        x = x.mean(dim=1)               # Global Average Pooling
        return self.classifier(x)


model = StockTransformer(d_model=D_MODEL, nhead=4, num_layers=2, seq_len=SEQ_LEN)
total_params = sum(p.numel() for p in model.parameters())
print(f"   → 총 파라미터: {total_params:,}개")
time.sleep(0.4)

# ── 5. 학습 설정 ───────────────────────────────────────────
print("\n[5/8] 학습 설정 중 (CrossEntropy + Adam + WarmupCosine)...")
time.sleep(0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
EPOCHS, BATCH = 120, 32
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
print(f"   → 에폭={EPOCHS}  배치={BATCH}  d_model={D_MODEL}  nhead=4")
time.sleep(0.3)

# ── 6. 학습 루프 ──────────────────────────────────────────
print("\n[6/8] Transformer 학습 시작...")
print("   Multi-Head Self-Attention이 날짜 간 상관관계를 학습합니다")
time.sleep(0.8)

loss_hist, acc_hist = [], []
attn_weights_sample = None
prev_loss = None
model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    ep_loss, ep_correct = 0.0, 0
    for s in range(0, len(X_train), BATCH):
        idx = perm[s:s + BATCH]
        xb, yb = X_train[idx], y_train[idx]
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss    += loss.item() * len(xb)
        ep_correct += (out.argmax(1) == yb).sum().item()
    scheduler.step()
    avg_loss = ep_loss / len(X_train)
    avg_acc  = ep_correct / len(X_train)
    loss_hist.append(avg_loss)
    acc_hist.append(avg_acc)
    if epoch % 30 == 0:
        trend = " ↓" if (prev_loss and avg_loss < prev_loss) else ""
        print(f"   Epoch {epoch:4d} | Loss: {avg_loss:.4f}{trend}  Acc: {avg_acc:.4f}")
        prev_loss = avg_loss
        time.sleep(0.2)

print(f"   Epoch {EPOCHS:4d} | 학습 완료!")
time.sleep(0.4)

# ── 7. 테스트 평가 & Attention 가중치 추출 ─────────────────
print("\n[7/8] 테스트 평가 & Attention 가중치 시각화 준비 중...")
time.sleep(0.4)
model.eval()
with torch.no_grad():
    out_test = model(X_test)
    preds    = out_test.argmax(1)
    acc_test = (preds == y_test).float().mean().item()
    probs    = torch.softmax(out_test, dim=1)[:, 1].numpy()
print(f"   → 테스트 정확도: {acc_test:.4f} ({acc_test * 100:.1f}%)")

# Attention 가중치 추출 (첫 번째 레이어, 첫 번째 샘플)
sample = X_test[:1]   # (1, 30, 1)
with torch.no_grad():
    x_proj = model.input_proj(sample)
    x_enc  = model.pos_enc(x_proj)
    # TransformerEncoderLayer의 self_attn에서 가중치 추출
    first_layer = model.encoder.layers[0]
    _, attn_w = first_layer.self_attn(x_enc, x_enc, x_enc, need_weights=True)
    attn_map = attn_w[0].numpy()   # (seq_len, seq_len)
time.sleep(0.3)

# ── 8. 시각화 ─────────────────────────────────────────────
print("\n[8/8] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(14, 10))

# 학습 손실
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(loss_hist, color='steelblue')
ax1.set_title("학습 손실")
ax1.set_xlabel("에폭")
ax1.set_ylabel("손실")
ax1.grid(alpha=0.3)

# 학습 정확도
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(acc_hist, color='tomato')
ax2.axhline(0.5, linestyle='--', color='gray', linewidth=0.8, label='무작위=0.5')
ax2.set_title(f"학습 정확도  |  테스트={acc_test:.2f}")
ax2.set_xlabel("에폭")
ax2.set_ylabel("정확도")
ax2.legend()
ax2.grid(alpha=0.3)

# 테스트 예측 확률
ax3 = fig.add_subplot(2, 3, 3)
n_show = min(60, len(probs))
colors = ['tomato' if p >= 0.5 else 'royalblue' for p in probs[:n_show]]
ax3.bar(range(n_show), probs[:n_show], color=colors, alpha=0.8, edgecolor='k', linewidth=0.2)
ax3.axhline(0.5, linestyle='--', color='black', linewidth=0.8)
ax3.set_title(f"상승 예측 확률 (정확도={acc_test:.2f})")
ax3.set_xlabel("테스트 샘플")
ax3.set_ylabel("상승 확률")

# Attention 히트맵
ax4 = fig.add_subplot(2, 1, 2)
im = ax4.imshow(attn_map, cmap='hot', aspect='auto')
plt.colorbar(im, ax=ax4, fraction=0.02)
ax4.set_title("Self-Attention 가중치 히트맵 (첫 번째 레이어, 테스트 샘플 1개)\n"
              "밝을수록 해당 날짜끼리 서로 높은 Attention 가중치를 가짐")
ax4.set_xlabel("Key 위치 (날짜)")
ax4.set_ylabel("Query 위치 (날짜)")

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         "Transformer — '어느 날'이 예측에 중요한지 스스로 찾아서 집중합니다 (Attention)",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 손실 패널 (ax1) ──
ax1.annotate('처음엔 많이 틀려요', xy=(0.0, 0.95), xytext=(0.08, 0.82),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax1.annotate('점점 배워요', xy=(0.9, 0.25), xytext=(0.60, 0.45),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')

# ── 정확도 패널 (ax2) ──
ax2.text(0.5, -0.18,
         '동전 던지기 수준(0.5)보다 높아야 의미 있어요',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')

# ── 예측확률 패널 (ax3) ──
ax3.text(0.5, -0.18,
         '0.5 위=상승 예측, 0.5 아래=하락 예측',
         transform=ax3.transAxes, ha='center', fontsize=7, color='gray')

# ── Attention 히트맵 패널 (ax4) — 가장 중요 ──
ax4.text(0.5, 1.03,
         "밝은 칸 = 두 날짜가 서로 '관련 있다'고 AI가 판단한 것",
         transform=ax4.transAxes, ha='center', fontsize=8, color='#333')
ax4.text(0.5, -0.10,
         '참고하는 날짜 위치',
         transform=ax4.transAxes, ha='center', fontsize=7, color='gray')
ax4.set_ylabel('현재 보고 있는 날짜')
ax4.annotate('가까운 날짜끼리 서로 주목해요', xy=(0.35, 0.65), xytext=(0.55, 0.82),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='white'), fontsize=7, color='white')
ax4.text(0.5, 0.08,
         '마치 책을 읽을 때 중요한 단어에 밑줄 긋는 것처럼,\nAI가 중요한 날짜에 집중해요',
         transform=ax4.transAxes, ha='center', fontsize=7.5, color='white',
         bbox=dict(boxstyle='round', facecolor='#333', alpha=0.55))

plt.subplots_adjust(top=0.93)
plt.tight_layout()
ticker_tag = TICKER.replace(".", "_")
out_name = f"../result/TransformerAttention_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ Transformer Self-Attention 주가 방향 분류 실습 완료!\n")
