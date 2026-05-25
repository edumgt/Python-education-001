import os
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 65)
print("  DQN 강화학습 트레이딩 에이전트 실습")
print("=" * 65)
print()
print("  강화학습 핵심 개념:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  에이전트(Agent)  : 매수/매도/관망을 결정하는 AI     │")
print("  │  환경(Env)        : 주식 시장 (가격 변동)            │")
print("  │  상태(State)      : 최근 N일 수익률 + 보유 여부      │")
print("  │  행동(Action)     : 0=관망  1=매수  2=매도           │")
print("  │  보상(Reward)     : 매도 시 실현 수익률              │")
print("  │  Q함수(DQN)       : 상태+행동 → 기대 누적 보상 예측  │")
print("  │                                                       │")
print("  │  핵심 기법                                            │")
print("  │  ε-탐욕: 처음엔 무작위, 점차 학습된 판단 사용       │")
print("  │  경험 재현(Replay): 과거 경험을 섞어 편향 제거       │")
print("  │  타깃 네트워크: 학습 안정화를 위한 고정 복사본       │")
print("  └──────────────────────────────────────────────────────┘")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
DEVICE = torch.device("cpu")

# ── 1. 주가 데이터 로드 ───────────────────────────────────────
print("\n[1/9] 주가 데이터 로드 중 (078935.KS - GS피앤엘)...")
time.sleep(0.5)

TICKER = '078935.KS'
prices = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2018-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 100:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    days = 700
    t = np.arange(days, dtype=float)
    prices = (10000 + 50 * t + 800 * np.sin(t / 40)
              + 300 * np.sin(t / 10) + np.random.normal(0, 150, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

returns = np.diff(prices) / prices[:-1]          # 일별 수익률
print(f"   → {len(prices)}일치 주가  |  수익률: {returns.min():.3f} ~ {returns.max():.3f}")
time.sleep(0.3)

# ── 2. 트레이딩 환경 정의 ─────────────────────────────────────
print("\n[2/9] 트레이딩 환경(Environment) 정의 중...")
print("   상태(State) = 최근 20일 수익률(20차원) + 현재 포지션(1차원) = 21차원")
time.sleep(0.5)

SEQ_LEN  = 20    # 관측 윈도우
N_ACTS   = 3     # 0=관망 1=매수 2=매도
TRAIN_SPLIT = 0.8


class TradingEnv:
    """단순 롱-온리 환경: 주식을 1주 매수하거나 매도하거나 관망"""
    def __init__(self, ret_series):
        self.rets  = ret_series
        self.n     = len(ret_series)
        self.reset()

    def reset(self):
        self.t        = SEQ_LEN          # 현재 날짜 인덱스
        self.position = 0                # 0=미보유  1=보유
        self.entry    = 0.0              # 매수 시점 수익률 인덱스
        return self._obs()

    def _obs(self):
        window = self.rets[self.t - SEQ_LEN:self.t]
        # z-score 정규화로 스케일 중립화
        mu, sd = window.mean(), window.std() + 1e-8
        norm   = (window - mu) / sd
        return np.append(norm, float(self.position)).astype(np.float32)

    def step(self, action):
        r = self.rets[self.t]            # 오늘 실현될 수익률

        reward = 0.0
        if action == 1 and self.position == 0:   # 매수
            self.position = 1
            self.entry    = r
        elif action == 2 and self.position == 1:  # 매도
            self.position = 0
            reward = r - 0.001           # 거래 비용 0.1%
        elif self.position == 1:                  # 보유 중 관망 → 소액 보상
            reward = r * 0.1

        self.t += 1
        done = self.t >= self.n - 1
        return self._obs(), reward, done


split_idx = int(len(returns) * TRAIN_SPLIT)
train_env = TradingEnv(returns[:split_idx])
test_env  = TradingEnv(returns[split_idx:])
STATE_DIM = SEQ_LEN + 1
print(f"   → 학습 환경: {split_idx}일  |  테스트 환경: {len(returns)-split_idx}일")
time.sleep(0.3)

# ── 3. DQN 모델 정의 ──────────────────────────────────────────
print("\n[3/9] DQN(Deep Q-Network) 모델 정의 중...")
print("   구조: FC(21→128) → ReLU → FC(128→64) → ReLU → FC(64→3)")
print("   출력: 각 행동(관망/매수/매도)의 Q값(기대 누적 보상)")
time.sleep(0.5)


class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


policy_net = DQN().to(DEVICE)
target_net = DQN().to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
total_params = sum(p.numel() for p in policy_net.parameters())
print(f"   → 총 파라미터: {total_params:,}개  (policy + target 각 동일)")
time.sleep(0.4)

# ── 4. 학습 하이퍼파라미터 & 경험 재현 버퍼 ───────────────────
print("\n[4/9] 하이퍼파라미터 & 경험 재현 버퍼 설정 중...")
time.sleep(0.4)

GAMMA        = 0.95     # 미래 보상 할인율
LR           = 1e-3
BATCH_SIZE   = 64
BUFFER_CAP   = 5_000   # 경험 재현 버퍼 최대 크기
TARGET_UPDATE = 20      # 에피소드 단위 타깃 네트워크 업데이트 주기
EPISODES     = 80
EPS_START    = 1.0      # ε-탐욕 초기값 (완전 탐색)
EPS_END      = 0.05     # ε-탐욕 최솟값
EPS_DECAY    = 0.97     # 에피소드마다 감소 비율

replay_buffer = deque(maxlen=BUFFER_CAP)
optimizer     = torch.optim.Adam(policy_net.parameters(), lr=LR)
criterion     = nn.SmoothL1Loss()   # Huber Loss: 이상치에 강인

print(f"   → 에피소드={EPISODES}  배치={BATCH_SIZE}  γ={GAMMA}")
print(f"   → ε: {EPS_START} → {EPS_END} (×{EPS_DECAY}/에피소드)")
print(f"   → 버퍼={BUFFER_CAP}  타깃 업데이트={TARGET_UPDATE}에피소드마다")
time.sleep(0.3)

# ── 5. ε-탐욕 행동 선택 함수 ─────────────────────────────────
print("\n[5/9] ε-탐욕(Epsilon-Greedy) 전략 정의 중...")
print("   처음엔 ε=1.0 → 무작위 행동(탐색)")
print("   학습할수록 ε 감소 → Q값 기반 최적 행동(활용)")
time.sleep(0.5)


def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTS - 1)          # 무작위 탐색
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return policy_net(s).argmax().item()           # 최대 Q값 행동


def optimize():
    if len(replay_buffer) < BATCH_SIZE:
        return None
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    s  = torch.tensor(np.array(states),      dtype=torch.float32).to(DEVICE)
    a  = torch.tensor(actions,               dtype=torch.long).to(DEVICE)
    r  = torch.tensor(rewards,               dtype=torch.float32).to(DEVICE)
    ns = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
    d  = torch.tensor(dones,                 dtype=torch.float32).to(DEVICE)

    q_val  = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)   # 현재 Q
    with torch.no_grad():
        q_next = target_net(ns).max(1)[0]                          # 타깃 Q
    q_target = r + GAMMA * q_next * (1 - d)                        # Bellman

    loss = criterion(q_val, q_target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# ── 6. DQN 학습 루프 ─────────────────────────────────────────
print("\n[6/9] DQN 학습 시작...")
print("   에피소드마다 전체 학습 기간 순서대로 시뮬레이션")
time.sleep(0.8)

ep_rewards, ep_losses, ep_epsilons = [], [], []
epsilon = EPS_START

for ep in range(1, EPISODES + 1):
    state    = train_env.reset()
    ep_reward = 0.0
    ep_loss_sum, ep_steps = 0.0, 0

    while True:
        action           = select_action(state, epsilon)
        next_state, reward, done = train_env.step(action)
        replay_buffer.append((state, action, reward, next_state, float(done)))
        state             = next_state
        ep_reward        += reward

        loss = optimize()
        if loss is not None:
            ep_loss_sum += loss
            ep_steps    += 1

        if done:
            break

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # 타깃 네트워크 업데이트
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    avg_loss = ep_loss_sum / max(ep_steps, 1)
    ep_rewards.append(ep_reward)
    ep_losses.append(avg_loss)
    ep_epsilons.append(epsilon)

    if ep % 20 == 0 or ep == 1:
        print(f"   Ep {ep:4d} | 누적보상={ep_reward:+.4f}  "
              f"손실={avg_loss:.5f}  ε={epsilon:.3f}")
        time.sleep(0.2)

print("   학습 완료!")
time.sleep(0.4)

# ── 7. 테스트 세트 백테스트 ──────────────────────────────────
print("\n[7/9] 테스트 데이터 백테스트 중...")
time.sleep(0.4)

policy_net.eval()
state     = test_env.reset()
portfolio = [1.0]          # 초기 자산 = 1 (정규화)
actions_log = []

with torch.no_grad():
    while True:
        s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action = policy_net(s_t).argmax().item()
        next_state, reward, done = test_env.step(action)
        actions_log.append(action)

        # 단순 Buy&Hold 비교를 위해 포트폴리오 추적
        r_today = test_env.rets[test_env.t - 1]
        if test_env.position == 1:
            portfolio.append(portfolio[-1] * (1 + r_today))
        else:
            portfolio.append(portfolio[-1])

        state = next_state
        if done:
            break

# Buy & Hold 수익률 (테스트 구간)
bh_returns = returns[split_idx:]
bh_curve   = np.cumprod(1 + bh_returns)

dqn_total  = (portfolio[-1] - 1) * 100
bh_total   = (bh_curve[-1]  - 1) * 100
act_cnt    = {0: actions_log.count(0), 1: actions_log.count(1), 2: actions_log.count(2)}
print(f"   → DQN 수익률:       {dqn_total:+.2f}%")
print(f"   → Buy & Hold 수익률: {bh_total:+.2f}%")
print(f"   → 행동 분포: 관망={act_cnt[0]}  매수={act_cnt[1]}  매도={act_cnt[2]}")
time.sleep(0.3)

# ── 8. 학습 진행 분석 ─────────────────────────────────────────
print("\n[8/9] 학습 진행 분석 중...")
time.sleep(0.3)

window_size = 10
smoothed_rewards = np.convolve(ep_rewards, np.ones(window_size)/window_size, mode='valid')
print(f"   → 초기 10회 평균 보상: {np.mean(ep_rewards[:10]):.4f}")
print(f"   → 최종 10회 평균 보상: {np.mean(ep_rewards[-10:]):.4f}")
time.sleep(0.3)

# ── 9. 시각화 ─────────────────────────────────────────────────
print("\n[9/9] 시각화 저장 중...")
time.sleep(0.5)

fig = plt.figure(figsize=(16, 12))

fig.text(0.5, 0.98,
         "DQN 강화학습 트레이딩 — 주식 시장과의 상호작용으로 매매 전략을 스스로 학습합니다",
         ha='center', fontsize=10, color='#333', weight='bold')

# 에피소드 보상
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(ep_rewards, color='steelblue', alpha=0.4, linewidth=1, label='에피소드 보상')
if len(smoothed_rewards) > 0:
    x_smooth = range(window_size - 1, len(ep_rewards))
    ax1.plot(list(x_smooth), smoothed_rewards, color='navy', linewidth=2, label=f'{window_size}회 이동평균')
ax1.axhline(0, linestyle='--', color='gray', linewidth=0.8)
ax1.set_title("에피소드 누적 보상")
ax1.set_xlabel("에피소드")
ax1.set_ylabel("보상")
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3)
ax1.annotate('초기: 무작위 탐색', xy=(0, ep_rewards[0]), xytext=(5, ep_rewards[0] + 0.02),
             fontsize=7, color='darkred',
             arrowprops=dict(arrowstyle='->', color='gray'))

# 학습 손실
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(ep_losses, color='tomato', linewidth=1.2)
ax2.set_title("에피소드 평균 Q-손실 (Huber Loss)")
ax2.set_xlabel("에피소드")
ax2.set_ylabel("손실")
ax2.grid(alpha=0.3)
ax2.text(0.5, -0.18, 'Q값 예측 오차 — 낮을수록 Q함수가 정확합니다',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')

# ε 감소 곡선
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(ep_epsilons, color='purple', linewidth=1.5)
ax3.set_title("ε (탐색률) 감소 곡선")
ax3.set_xlabel("에피소드")
ax3.set_ylabel("ε 값")
ax3.grid(alpha=0.3)
ax3.annotate('탐색(무작위)', xy=(0, EPS_START), xytext=(5, EPS_START - 0.1),
             fontsize=7, color='purple',
             arrowprops=dict(arrowstyle='->', color='gray'))
ax3.annotate(f'활용(학습된 Q)\nε={EPS_END}', xy=(EPISODES-1, EPS_END),
             xytext=(EPISODES * 0.6, EPS_END + 0.15),
             fontsize=7, color='navy',
             arrowprops=dict(arrowstyle='->', color='gray'))

# 테스트 포트폴리오 vs Buy & Hold
ax4 = fig.add_subplot(3, 1, 2)
t_axis = range(len(portfolio))
ax4.plot(t_axis, portfolio, color='tomato', linewidth=1.8, label='DQN 에이전트')
bh_aligned = np.concatenate([[1.0], bh_curve])[:len(portfolio)]
ax4.plot(range(len(bh_aligned)), bh_aligned, color='steelblue', linewidth=1.5,
         linestyle='--', label='Buy & Hold')
ax4.axhline(1.0, linestyle=':', color='gray', linewidth=0.8)
ax4.set_title(f"테스트 구간 포트폴리오 성과  |  DQN={dqn_total:+.2f}%  B&H={bh_total:+.2f}%")
ax4.set_xlabel("거래일 (테스트)")
ax4.set_ylabel("자산 (초기=1.0)")
ax4.legend()
ax4.grid(alpha=0.3)
ax4.text(0.5, -0.12,
         'DQN이 Buy&Hold보다 높으면 하락장 회피 등 부가가치를 창출한 것',
         transform=ax4.transAxes, ha='center', fontsize=7.5, color='gray')

# 행동 분포 파이차트
ax5 = fig.add_subplot(3, 3, 7)
labels  = ['관망', '매수', '매도']
counts  = [act_cnt[0], act_cnt[1], act_cnt[2]]
colors5 = ['#a8d8a8', '#f08080', '#87ceeb']
wedges, texts, autotexts = ax5.pie(
    counts, labels=labels, colors=colors5, autopct='%1.1f%%', startangle=90)
for at in autotexts:
    at.set_fontsize(8)
ax5.set_title("테스트 행동 분포")

# Q값 분포 (테스트 첫 50 스텝)
ax6 = fig.add_subplot(3, 3, 8)
policy_net.eval()
test_env2 = TradingEnv(returns[split_idx:])
state2    = test_env2.reset()
q_samples = []
with torch.no_grad():
    for _ in range(min(100, len(returns) - split_idx - SEQ_LEN - 1)):
        s_t = torch.tensor(state2, dtype=torch.float32).unsqueeze(0)
        q   = policy_net(s_t).squeeze().numpy()
        q_samples.append(q)
        action2 = int(np.argmax(q))
        state2, _, done2 = test_env2.step(action2)
        if done2:
            break
q_arr = np.array(q_samples)
for i, (lbl, col) in enumerate(zip(labels, colors5)):
    ax6.plot(q_arr[:, i], label=f'Q({lbl})', linewidth=1.2, color=col)
ax6.set_title("Q값 시간 변화 (테스트 초반 100스텝)")
ax6.set_xlabel("스텝")
ax6.set_ylabel("Q값")
ax6.legend(fontsize=7)
ax6.grid(alpha=0.3)

# 구조 설명 텍스트
ax7 = fig.add_subplot(3, 3, 9)
ax7.axis('off')
struct = (
    "DQN 학습 흐름\n\n"
    "① 상태 관측\n"
    "   (최근 20일 수익률 + 포지션)\n\n"
    "② ε-탐욕으로 행동 선택\n"
    "   (관망/매수/매도)\n\n"
    "③ 환경 → 보상 수신\n\n"
    "④ 경험 버퍼에 저장\n"
    "   (s, a, r, s', done)\n\n"
    "⑤ 미니배치 샘플링\n"
    "   Bellman 방정식으로 학습\n\n"
    "⑥ TARGET 네트워크 주기적 동기화\n\n"
    f"버퍼크기: {BUFFER_CAP:,}\n"
    f"γ(할인율): {GAMMA}\n"
    f"총 에피소드: {EPISODES}"
)
ax7.text(0.05, 0.95, struct, transform=ax7.transAxes,
         fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.subplots_adjust(top=0.94, hspace=0.45, wspace=0.35)
ticker_tag = TICKER.replace('.', '_')
out_name   = f"../result/DqnTradingAgent_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ DQN 강화학습 트레이딩 에이전트 실습 완료!\n")
