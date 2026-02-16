"""
==========================================================================
實驗設計腳本 - 基於會議紀錄 (2026/1/23)
==========================================================================
研究目的: 改變固定參數 (θ, hf, s, c) 的值，觀察如何影響廠商三大決策:
    1. BOPS 策略 (是否開啟 Buy Online Pick-up in Store)
    2. 定價策略 (產品定價)
    3. φ (phi) 補貼率

實驗方式: 敏感度分析 (Sensitivity Analysis)
    - 每次僅改變一個參數，其餘固定於基準值
    - 對每組參數值訓練 MADDPG 模型至均衡
    - 記錄均衡時兩家廠商的決策結果
==========================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import os
import json
import time
from itertools import product

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 0. Device Configuration
# ==============================================================================
device = torch.device("cpu")

# ==============================================================================
# 1. 基準參數 (Baseline Parameters)
# ==============================================================================
BASELINE_PARAMS = {
    'PARAM_V': 170.0,        # 產品保留價值 v
    'PARAM_S': 25.0,         # 運費 s
    'PARAM_T': 15.0,         # 線上等待成本 t
    'PARAM_MC': 30.0,        # 邊際成本
    'PARAM_C': 2.0,          # BOPS 運營成本 c
    'FIXED_COST_BOPS': 5.0,  # BOPS 固定成本
    'PARAM_T_HOTELLING': 20.0,  # Hotelling 品牌偏好係數
    'PARAM_THETA': 0.8,      # misfit cost 折扣因子 θ
    'PARAM_BETA': 0.3,       # Store-only 消費者比例
    'PARAM_ALPHA': 0.5,      # Low travel cost 比例
    'PARAM_H_L': 2.0,        # 低拜訪成本 hf_L
    'PARAM_H_H': 50.0,       # 高拜訪成本 hf_H
    'MAX_PRICE': 170.0,
    'NUM_CONSUMERS': 1000,
}

# ==============================================================================
# 2. 實驗參數設計 (Experiment Parameter Grid)
# ==============================================================================
EXPERIMENT_CONFIGS = {
    # θ 越大 → 線上 misfit = θ·t_H·d 越大 → 線上優勢越小
    'theta': {
        'param_name': 'PARAM_THETA',
        'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'label': 'θ (Theta)',
        'description': 'Misfit cost 折扣因子: θ 越大，線上 misfit 越接近實體（線上優勢越小）；θ 越小，線上購物不確定性損失越低',
    },
    'hf': {
        'param_name': 'PARAM_H_H',
        'values': [5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0],
        'label': 'hf (高拜訪成本)',
        'description': '消費者拜訪實體店面成本: hf 越高，消費者越傾向線上購買',
    },
    's': {
        'param_name': 'PARAM_S',
        'values': [5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 60.0],
        'label': 's (運費)',
        'description': '運費: s 越高，消費者線上購物成本越高，可能促進 BOPS 採用',
    },
    'c': {
        'param_name': 'PARAM_C',
        'values': [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0],
        'label': 'c (BOPS 運營成本)',
        'description': 'BOPS 運營管理成本: c 越高，廠商開啟 BOPS 的意願越低',
    },
}

# ==============================================================================
# 3. 超參數 (Training Hyperparameters)
# ==============================================================================
TRAIN_CONFIG = {
    'learning_rate_actor': 0.0001,
    'learning_rate_critic': 0.001,
    'discount_factor': 0.95,
    'tau': 0.005,
    'num_episodes': 4000,
    'batch_size': 256,
    'buffer_size': 200000,
    'max_steps_per_episode': 10,
    'exploration_noise_start': 0.5,
    'noise_decay': 0.9992,
    'min_exploration_noise': 0.01,
    'NUM_AGENTS': 2,
    'ACTION_SIZE': 3,           # (Price, BOPS, Phi)
    # 收斂檢測
    'convergence_window': 300,  # 檢查最後 N 回合
    'convergence_threshold': 0.005,  # reward std < 此值視為收斂
    'min_episodes': 2000,       # 至少跑這麼多回合才開始檢測收斂
    # 重複實驗
    'num_seeds': 3,             # 每組參數重複幾次取均值 (設 1 = 只跑一次)
}

GLOBAL_STATE_SIZE = TRAIN_CONFIG['ACTION_SIZE'] * TRAIN_CONFIG['NUM_AGENTS']
TOTAL_ACTION_SIZE = TRAIN_CONFIG['ACTION_SIZE'] * TRAIN_CONFIG['NUM_AGENTS']
ACTION_SIZE = TRAIN_CONFIG['ACTION_SIZE']
NUM_AGENTS = TRAIN_CONFIG['NUM_AGENTS']

# ==============================================================================
# 4. 網路定義
# ==============================================================================
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return torch.sigmoid(self.fc3(x))


class CriticNetwork(nn.Module):
    def __init__(self, state_size, total_action_size):
        super().__init__()
        self.fc_s = nn.Linear(state_size, 64)
        self.fc_a = nn.Linear(total_action_size, 64)
        self.fc_cat = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, state, action):
        s = F.relu(self.fc_s(state))
        a = F.relu(self.fc_a(action))
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc_cat(cat))
        return self.fc_out(x)


# ==============================================================================
# 5. Agent & Replay Buffer
# ==============================================================================
class Agent:
    def __init__(self, agent_id, state_size, action_size, total_action_size, lr_actor, lr_critic):
        self.agent_id = agent_id
        self.action_size = action_size
        self.actor = ActorNetwork(state_size, action_size).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = CriticNetwork(state_size, total_action_size).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def update_target_networks(self, tau):
        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def select_action(self, state, noise_std):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            raw = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        noise = np.random.normal(0, noise_std, size=self.action_size)
        return np.clip(raw + noise, 0.0, 1.0)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, actions, rewards, next_state):
        self.buffer.append((state, actions, rewards, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# 6. MADDPG Trainer
# ==============================================================================
class MADDPGTrainer:
    def __init__(self, num_agents, state_size, action_size, total_action_size, cfg):
        self.num_agents = num_agents
        self.agents = [
            Agent(i, state_size, action_size, total_action_size,
                  cfg['learning_rate_actor'], cfg['learning_rate_critic'])
            for i in range(num_agents)
        ]
        self.buffer = ReplayBuffer(cfg['buffer_size'])
        self.train_step_count = 0

    def get_actions(self, state, noise_std):
        return np.array([a.select_action(state, noise_std) for a in self.agents])

    def store_experience(self, state, actions, rewards, next_state):
        self.buffer.add(state, actions, rewards, next_state)

    def train(self, batch_size, discount_factor, tau, params):
        if len(self.buffer) < batch_size:
            return
        self.train_step_count += 1
        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        states_t = torch.FloatTensor(states).to(device)
        raw_actions_t = torch.FloatTensor(actions).view(batch_size, -1).to(device)
        actions_t = scale_action_torch(raw_actions_t, params)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)

        with torch.no_grad():
            target_next = []
            for agent in self.agents:
                pred = agent.target_actor(next_states_t)
                target_next.append(scale_action_torch(pred, params))
            target_next_t = torch.cat(target_next, dim=1)

        for i, agent in enumerate(self.agents):
            agent_rewards = rewards_t[:, i].unsqueeze(1)
            with torch.no_grad():
                target_q = agent.target_critic(next_states_t, target_next_t)
                targets = agent_rewards + discount_factor * target_q
            current_q = agent.critic(states_t, actions_t)
            critic_loss = F.mse_loss(current_q, targets)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            if self.train_step_count % 2 == 0:
                preds_all = []
                for j, other in enumerate(self.agents):
                    pred = other.actor(states_t)
                    scaled_pred = scale_action_torch(pred, params)
                    if j != i:
                        scaled_pred = scaled_pred.detach()
                    preds_all.append(scaled_pred)
                preds_all_t = torch.cat(preds_all, dim=1)
                actor_loss = -agent.critic(states_t, preds_all_t).mean()
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor_optimizer.step()

        if self.train_step_count % 2 == 0:
            for agent in self.agents:
                agent.update_target_networks(tau)


# ==============================================================================
# 7. 環境函數
# ==============================================================================
def scale_action(action, params):
    """將 [0,1] 縮放回真實環境數值"""
    scaled = np.copy(action)
    min_price = params['PARAM_MC'] + 5.0
    scaled[0] = min_price + action[0] * (params['MAX_PRICE'] - min_price)
    return scaled


def scale_action_torch(action_tensor, params):
    min_price = params['PARAM_MC'] + 5.0
    scaled = action_tensor.clone()
    for i in range(scaled.shape[1] // ACTION_SIZE):
        idx = i * ACTION_SIZE
        scaled[:, idx:idx + 1] = min_price + action_tensor[:, idx:idx + 1] * (params['MAX_PRICE'] - min_price)
    return scaled


def calculate_profits(action1, action2, params):
    """計算利潤 (Hotelling 模型)"""
    p1, bops1, phi1 = action1
    p2, bops2, phi2 = action2

    k = 15.0
    w1 = 1.0 / (1.0 + np.exp(-k * (bops1 - 0.5)))
    w2 = 1.0 / (1.0 + np.exp(-k * (bops2 - 0.5)))

    V = params['PARAM_V']
    S = params['PARAM_S']
    T = params['PARAM_T']
    MC = params['PARAM_MC']
    C = params['PARAM_C']
    T_H = params['PARAM_T_HOTELLING']
    theta = params['PARAM_THETA']
    beta = params['PARAM_BETA']
    alpha = params['PARAM_ALPHA']
    H_L = params['PARAM_H_L']
    H_H = params['PARAM_H_H']
    FIXED = params['FIXED_COST_BOPS']
    N = params['NUM_CONSUMERS']

    x_loc = np.linspace(0, 1, N)
    dist1 = x_loc
    dist2 = 1 - x_loc

    def get_segment_choices(h_cost, force_store_only=False):
        u_s1 = V - p1 - T_H * dist1 - h_cost
        u_s2 = V - p2 - T_H * dist2 - h_cost

        if force_store_only:
            u_o1 = u_o2 = u_b1 = u_b2 = np.full_like(x_loc, -9999.0)
        else:
            s1_consumer = (1 - phi1) * S
            s2_consumer = (1 - phi2) * S
            u_o1 = V - p1 - theta * T_H * dist1 - s1_consumer - T
            u_o2 = V - p2 - theta * T_H * dist2 - s2_consumer - T
            bops_penalty1 = (1.0 - w1) * 200.0
            bops_penalty2 = (1.0 - w2) * 200.0
            u_b1 = V - p1 - theta * T_H * dist1 - h_cost - bops_penalty1
            u_b2 = V - p2 - theta * T_H * dist2 - h_cost - bops_penalty2

        u_no_buy = np.zeros_like(x_loc)
        all_u = np.vstack([u_s1, u_o1, u_b1, u_s2, u_o2, u_b2, u_no_buy])
        choices = np.argmax(all_u, axis=0)
        counts = np.zeros(7)
        for i in range(7):
            counts[i] = np.sum(choices == i) / N
        return counts[:6]

    d_beta = get_segment_choices(H_L, force_store_only=True)
    d_alpha_low = get_segment_choices(H_L, force_store_only=False)
    d_alpha_high = get_segment_choices(H_H, force_store_only=False)

    total = beta * d_beta + (1 - beta) * (alpha * d_alpha_low + (1 - alpha) * d_alpha_high)
    d_s1, d_o1, d_b1 = total[0], total[1], total[2]
    d_s2, d_o2, d_b2 = total[3], total[4], total[5]

    profit1 = (p1 - MC) * (d_s1 + d_o1 + d_b1) - phi1 * S * d_o1 - C * d_b1 - w1 * FIXED
    profit2 = (p2 - MC) * (d_s2 + d_o2 + d_b2) - phi2 * S * d_o2 - C * d_b2 - w2 * FIXED

    return profit1, profit2, [d_s1, d_o1, d_b1], [d_s2, d_o2, d_b2]


# ==============================================================================
# 8. 單次實驗訓練函數 (加入收斂檢測)
# ==============================================================================
def run_single_experiment(params, cfg, seed=None, verbose=False):
    """
    針對一組環境參數訓練 MADDPG 並回傳均衡結果。
    加入收斂檢測以提早停止。
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    maddpg = MADDPGTrainer(NUM_AGENTS, GLOBAL_STATE_SIZE, ACTION_SIZE, TOTAL_ACTION_SIZE, cfg)
    current_state = np.zeros(GLOBAL_STATE_SIZE)
    current_noise = cfg['exploration_noise_start']

    history_rewards = []
    # 記錄每回合的動作（用於取均值）
    history_actions = []
    converged_at = cfg['num_episodes']

    for episode in range(cfg['num_episodes']):
        step_rewards = np.zeros(NUM_AGENTS)

        with torch.no_grad():
            st = torch.FloatTensor(current_state).unsqueeze(0).to(device)
            initial_raw = [a.actor(st).cpu().numpy()[0] for a in maddpg.agents]
            prev_env = [scale_action(a, params) for a in initial_raw]
        prev_bops = np.array([1 if a[1] > 0.5 else 0 for a in initial_raw])

        for step in range(cfg['max_steps_per_episode']):
            raw_actions = maddpg.get_actions(current_state, current_noise)
            env_actions = [scale_action(a, params) for a in raw_actions]

            r1, r2, _, _ = calculate_profits(env_actions[0], env_actions[1], params)

            price_pen_w = 5.0
            p1_jump = abs(env_actions[0][0] - prev_env[0][0]) / params['MAX_PRICE']
            p2_jump = abs(env_actions[1][0] - prev_env[1][0]) / params['MAX_PRICE']
            p1_pen = (p1_jump ** 2) * price_pen_w
            p2_pen = (p2_jump ** 2) * price_pen_w

            reward_scale = 100.0
            reward1 = (r1 / reward_scale) - p1_pen
            reward2 = (r2 / reward_scale) - p2_pen

            curr_bops = np.array([1 if raw_actions[0][1] > 0.5 else 0,
                                  1 if raw_actions[1][1] > 0.5 else 0])
            if curr_bops[0] != prev_bops[0]:
                reward1 -= 0.2
            if curr_bops[1] != prev_bops[1]:
                reward2 -= 0.2

            rewards = np.array([reward1, reward2], dtype=np.float32)
            prev_env = env_actions
            prev_bops = curr_bops

            next_state = np.concatenate(raw_actions)
            maddpg.store_experience(current_state, raw_actions, rewards, next_state)
            maddpg.train(cfg['batch_size'], cfg['discount_factor'], cfg['tau'], params)
            current_state = next_state
            step_rewards += rewards

        avg_rewards = step_rewards / cfg['max_steps_per_episode']
        history_rewards.append(avg_rewards)

        # 記錄無雜訊 actor 輸出
        with torch.no_grad():
            st = torch.FloatTensor(current_state).unsqueeze(0).to(device)
            clean_raw = [a.actor(st).cpu().numpy()[0] for a in maddpg.agents]
        clean_env = [scale_action(a, params) for a in clean_raw]
        history_actions.append(clean_env)

        current_noise = max(cfg['min_exploration_noise'], current_noise * cfg['noise_decay'])

        if verbose and (episode + 1) % 500 == 0:
            ea = clean_env
            print(f"  Ep {episode + 1} | "
                  f"F1: P={ea[0][0]:.1f}, B={int(ea[0][1] > 0.5)}, φ={ea[0][2]:.2f} | "
                  f"F2: P={ea[1][0]:.1f}, B={int(ea[1][1] > 0.5)}, φ={ea[1][2]:.2f}")

        # 收斂檢測
        conv_win = cfg.get('convergence_window', 300)
        conv_thresh = cfg.get('convergence_threshold', 0.005)
        min_ep = cfg.get('min_episodes', 2000)
        if episode >= min_ep and len(history_rewards) >= conv_win:
            recent = np.array(history_rewards[-conv_win:])
            if np.std(recent[:, 0]) < conv_thresh and np.std(recent[:, 1]) < conv_thresh:
                converged_at = episode + 1
                if verbose:
                    print(f"  ✓ 收斂於 Episode {converged_at}")
                break

    # --- 取最後 200 回合的均值作為均衡結果 ---
    avg_window = min(200, len(history_actions))
    recent_actions = history_actions[-avg_window:]

    eq_p1 = np.mean([a[0][0] for a in recent_actions])
    eq_bops1 = np.mean([a[0][1] for a in recent_actions])
    eq_phi1 = np.mean([a[0][2] for a in recent_actions])
    eq_p2 = np.mean([a[1][0] for a in recent_actions])
    eq_bops2 = np.mean([a[1][1] for a in recent_actions])
    eq_phi2 = np.mean([a[1][2] for a in recent_actions])

    eq_action1 = [eq_p1, eq_bops1, eq_phi1]
    eq_action2 = [eq_p2, eq_bops2, eq_phi2]
    profit1, profit2, demand1, demand2 = calculate_profits(eq_action1, eq_action2, params)

    # 收斂穩定性
    recent_rewards = np.array(history_rewards[-avg_window:])
    reward_std = np.std(recent_rewards, axis=0)

    result = {
        'firm1': {
            'price': eq_p1,
            'bops': int(eq_bops1 > 0.5),
            'bops_raw': eq_bops1,
            'phi': eq_phi1,
            'profit': profit1,
            'demand_store': demand1[0],
            'demand_online': demand1[1],
            'demand_bops': demand1[2],
        },
        'firm2': {
            'price': eq_p2,
            'bops': int(eq_bops2 > 0.5),
            'bops_raw': eq_bops2,
            'phi': eq_phi2,
            'profit': profit2,
            'demand_store': demand2[0],
            'demand_online': demand2[1],
            'demand_bops': demand2[2],
        },
        'convergence': {
            'converged_at': converged_at,
            'reward_std_f1': float(reward_std[0]),
            'reward_std_f2': float(reward_std[1]),
        },
        'history_rewards': np.array(history_rewards),
    }
    return result


# ==============================================================================
# 9. 實驗執行器 (支援多 seed 取均值)
# ==============================================================================
def run_experiment_suite(experiment_key, cfg=None, verbose=True):
    """
    針對指定實驗，依序跑不同參數值。
    支援 num_seeds > 1 時多次重複取均值。
    """
    if cfg is None:
        cfg = TRAIN_CONFIG
    exp_cfg = EXPERIMENT_CONFIGS[experiment_key]
    param_name = exp_cfg['param_name']
    values = exp_cfg['values']
    label = exp_cfg['label']
    num_seeds = cfg.get('num_seeds', 1)

    print(f"\n{'=' * 70}")
    print(f"實驗: {label} 的影響")
    print(f"描述: {exp_cfg['description']}")
    print(f"參數範圍: {values}")
    print(f"每組重複次數: {num_seeds}")
    print(f"{'=' * 70}")

    results = []
    for val in values:
        params = BASELINE_PARAMS.copy()
        params[param_name] = val
        print(f"\n--- {label} = {val} ---")
        t0 = time.time()

        if num_seeds == 1:
            result = run_single_experiment(params, cfg, verbose=verbose)
            result['param_value'] = val
            results.append(result)
        else:
            seed_results = []
            for s in range(num_seeds):
                print(f"  Seed {s + 1}/{num_seeds}...")
                r = run_single_experiment(params, cfg, seed=42 + s, verbose=False)
                seed_results.append(r)
            # 取均值
            avg_result = _average_seed_results(seed_results)
            avg_result['param_value'] = val
            # 保留第一次的 history_rewards 用於畫收斂曲線
            avg_result['history_rewards'] = seed_results[0]['history_rewards']
            results.append(avg_result)

        elapsed = time.time() - t0
        f1, f2 = results[-1]['firm1'], results[-1]['firm2']
        print(f"  結果 ({elapsed:.1f}s):")
        print(f"    Firm 1: Price={f1['price']:.1f}, BOPS={f1['bops']}, "
              f"φ={f1['phi']:.3f}, Profit={f1['profit']:.4f}")
        print(f"    Firm 2: Price={f2['price']:.1f}, BOPS={f2['bops']}, "
              f"φ={f2['phi']:.3f}, Profit={f2['profit']:.4f}")

    return results


def _average_seed_results(seed_results):
    """將多個 seed 的結果取均值"""
    keys = ['price', 'bops_raw', 'phi', 'profit',
            'demand_store', 'demand_online', 'demand_bops']
    result = {'firm1': {}, 'firm2': {}}
    for firm in ['firm1', 'firm2']:
        for k in keys:
            vals = [r[firm][k] for r in seed_results]
            result[firm][k] = float(np.mean(vals))
            result[firm][f'{k}_std'] = float(np.std(vals))
        result[firm]['bops'] = int(result[firm]['bops_raw'] > 0.5)
    return result


# ==============================================================================
# 10. 視覺化函數 (折線趨勢圖)
# ==============================================================================
def plot_experiment_results(results, experiment_key, save_dir='results'):
    """為單一實驗繪製折線趨勢圖 - 觀察參數如何影響各決策"""
    os.makedirs(save_dir, exist_ok=True)
    exp_cfg = EXPERIMENT_CONFIGS[experiment_key]
    label = exp_cfg['label']
    x_vals = [r['param_value'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'敏感度分析: {label} 對廠商決策的影響', fontsize=16, fontweight='bold')

    # 檢查是否有 std 資訊（多 seed）
    has_std = 'price_std' in results[0].get('firm1', {})

    def _plot_line(ax, x, y1, y2, ylabel, title, y1_std=None, y2_std=None):
        ax.plot(x, y1, 'o-', label='Firm 1', color='#2196F3', linewidth=2, markersize=6)
        ax.plot(x, y2, 's--', label='Firm 2', color='#F44336', linewidth=2, markersize=6)
        if y1_std is not None:
            ax.fill_between(x, np.array(y1) - np.array(y1_std),
                           np.array(y1) + np.array(y1_std), alpha=0.15, color='#2196F3')
            ax.fill_between(x, np.array(y2) - np.array(y2_std),
                           np.array(y2) + np.array(y2_std), alpha=0.15, color='#F44336')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # 1.1 定價策略
    y1 = [r['firm1']['price'] for r in results]
    y2 = [r['firm2']['price'] for r in results]
    std1 = [r['firm1'].get('price_std', 0) for r in results] if has_std else None
    std2 = [r['firm2'].get('price_std', 0) for r in results] if has_std else None
    _plot_line(axes[0, 0], x_vals, y1, y2, '均衡價格 (p)', '定價策略', std1, std2)

    # 1.2 BOPS 策略
    y1 = [r['firm1']['bops_raw'] for r in results]
    y2 = [r['firm2']['bops_raw'] for r in results]
    std1 = [r['firm1'].get('bops_raw_std', 0) for r in results] if has_std else None
    std2 = [r['firm2'].get('bops_raw_std', 0) for r in results] if has_std else None
    _plot_line(axes[0, 1], x_vals, y1, y2, 'BOPS 傾向值', 'BOPS 策略 (>0.5 為開啟)', std1, std2)
    axes[0, 1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    axes[0, 1].set_ylim(-0.05, 1.05)

    # 1.3 φ 補貼率
    y1 = [r['firm1']['phi'] for r in results]
    y2 = [r['firm2']['phi'] for r in results]
    std1 = [r['firm1'].get('phi_std', 0) for r in results] if has_std else None
    std2 = [r['firm2'].get('phi_std', 0) for r in results] if has_std else None
    _plot_line(axes[0, 2], x_vals, y1, y2, 'φ (補貼率)', '運費補貼率 φ', std1, std2)
    axes[0, 2].set_ylim(-0.05, 1.05)

    # 2.1 利潤
    y1 = [r['firm1']['profit'] for r in results]
    y2 = [r['firm2']['profit'] for r in results]
    std1 = [r['firm1'].get('profit_std', 0) for r in results] if has_std else None
    std2 = [r['firm2'].get('profit_std', 0) for r in results] if has_std else None
    _plot_line(axes[1, 0], x_vals, y1, y2, '均衡利潤 (π)', '廠商利潤', std1, std2)

    # 2.2 Firm 1 通路需求 (堆疊面積圖)
    ax = axes[1, 1]
    d_s = [r['firm1']['demand_store'] for r in results]
    d_o = [r['firm1']['demand_online'] for r in results]
    d_b = [r['firm1']['demand_bops'] for r in results]
    ax.stackplot(x_vals, d_s, d_o, d_b,
                 labels=['實體店', '線上', 'BOPS'],
                 colors=['#4CAF50', '#FF9800', '#9C27B0'], alpha=0.8)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('需求比例', fontsize=11)
    ax.set_title('Firm 1 通路需求分佈', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2.3 Firm 2 通路需求
    ax = axes[1, 2]
    d_s = [r['firm2']['demand_store'] for r in results]
    d_o = [r['firm2']['demand_online'] for r in results]
    d_b = [r['firm2']['demand_bops'] for r in results]
    ax.stackplot(x_vals, d_s, d_o, d_b,
                 labels=['實體店', '線上', 'BOPS'],
                 colors=['#4CAF50', '#FF9800', '#9C27B0'], alpha=0.8)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('需求比例', fontsize=11)
    ax.set_title('Firm 2 通路需求分佈', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'sensitivity_{experiment_key}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表已儲存: {save_path}")


def plot_convergence_check(results, experiment_key, save_dir='results'):
    """繪製各參數值下的訓練收斂曲線，驗證結果可靠性"""
    os.makedirs(save_dir, exist_ok=True)
    exp_cfg = EXPERIMENT_CONFIGS[experiment_key]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'{exp_cfg["label"]} - 訓練收斂驗證', fontsize=14, fontweight='bold')

    for r in results:
        rewards = r.get('history_rewards')
        if rewards is None:
            continue
        val = r['param_value']
        window = 100
        if len(rewards) >= window:
            smoothed_1 = np.convolve(rewards[:, 0], np.ones(window) / window, mode='valid')
            smoothed_2 = np.convolve(rewards[:, 1], np.ones(window) / window, mode='valid')
        else:
            smoothed_1, smoothed_2 = rewards[:, 0], rewards[:, 1]

        axes[0].plot(smoothed_1, label=f'{val}', alpha=0.8)
        axes[1].plot(smoothed_2, label=f'{val}', alpha=0.8)

    for i, title in enumerate(['Firm 1 Reward', 'Firm 2 Reward']):
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel('Smoothed Reward (MA-100)')
        axes[i].legend(title=exp_cfg['label'], fontsize=8)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'convergence_{experiment_key}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  收斂曲線已儲存: {save_path}")


def plot_all_experiments_comparison(all_results, save_dir='results'):
    """
    將所有實驗的結果放在同一張圖上做比較：
    每個子圖對應一個決策/結果變數，每條線對應一個實驗參數。
    → 方便一次看出哪個參數對哪個決策影響最大。
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ('price', '均衡價格 (p)'),
        ('bops_raw', 'BOPS 傾向值'),
        ('phi', '運費補貼率 φ'),
        ('profit', '廠商利潤 π'),
    ]
    colors = {'theta': '#2196F3', 'hf': '#F44336', 's': '#4CAF50', 'c': '#FF9800'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('所有實驗的綜合比較 (Firm 1)', fontsize=16, fontweight='bold')

    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for exp_key, results in all_results.items():
            # 正規化 x 軸到 [0, 1] 方便比較
            x_vals = [r['param_value'] for r in results]
            x_min, x_max = min(x_vals), max(x_vals)
            if x_max > x_min:
                x_norm = [(v - x_min) / (x_max - x_min) for v in x_vals]
            else:
                x_norm = x_vals
            y_vals = [r['firm1'][metric] for r in results]
            exp_label = EXPERIMENT_CONFIGS[exp_key]['label']
            ax.plot(x_norm, y_vals, 'o-', label=exp_label,
                    color=colors.get(exp_key, 'gray'), linewidth=2, markersize=5)

        ax.set_xlabel('正規化參數值 [0→最小, 1→最大]', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_all_experiments.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  綜合比較圖已儲存: {save_path}")


def plot_summary_table(all_results, save_dir='results'):
    """輸出摘要表"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 100}")
    print(f"{'實驗摘要表':^100}")
    print(f"{'=' * 100}")
    print(f"{'實驗':<12} {'參數值':<10} "
          f"{'F1 Price':<10} {'F1 BOPS':<9} {'F1 φ':<8} {'F1 Profit':<12} "
          f"{'F2 Price':<10} {'F2 BOPS':<9} {'F2 φ':<8} {'F2 Profit':<12}")
    print(f"{'-' * 100}")
    for exp_key, results in all_results.items():
        for r in results:
            f1, f2 = r['firm1'], r['firm2']
            print(f"{EXPERIMENT_CONFIGS[exp_key]['label']:<12} {r['param_value']:<10.1f} "
                  f"{f1['price']:<10.1f} {f1['bops']:<9d} {f1['phi']:<8.3f} {f1['profit']:<12.4f} "
                  f"{f2['price']:<10.1f} {f2['bops']:<9d} {f2['phi']:<8.3f} {f2['profit']:<12.4f}")
        print()


def save_results_json(all_results, save_dir='results'):
    """將結果存為 JSON"""
    os.makedirs(save_dir, exist_ok=True)
    save_data = {}
    for exp_key, results in all_results.items():
        exp_data = []
        for r in results:
            entry = {
                'param_value': r['param_value'],
                'firm1': {k: float(v) for k, v in r['firm1'].items()
                          if not isinstance(v, np.ndarray)},
                'firm2': {k: float(v) for k, v in r['firm2'].items()
                          if not isinstance(v, np.ndarray)},
            }
            if 'convergence' in r:
                entry['convergence'] = r['convergence']
            exp_data.append(entry)
        save_data[exp_key] = exp_data

    path = os.path.join(save_dir, 'experiment_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"結果已儲存: {path}")


# ==============================================================================
# 11. 主程式
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MADDPG 敏感度分析實驗")
    print("=" * 70)
    print(f"研究目的: 觀察 θ, hf, s, c 的變化如何影響廠商決策")
    print(f"觀察指標: BOPS策略 / 定價策略 / φ 補貼率 / 利潤")
    print(f"基準參數: θ={BASELINE_PARAMS['PARAM_THETA']}, "
          f"hf_H={BASELINE_PARAMS['PARAM_H_H']}, "
          f"s={BASELINE_PARAMS['PARAM_S']}, "
          f"c={BASELINE_PARAMS['PARAM_C']}")
    print(f"每組重複次數: {TRAIN_CONFIG['num_seeds']}")
    print("=" * 70)

    # --- 選擇要跑的實驗 ---
    # experiments_to_run = ['theta', 'hf', 's', 'c']  # 全部跑
    experiments_to_run = ['theta']  # 先跑一個驗證

    all_results = {}
    total_start = time.time()

    for exp_key in experiments_to_run:
        results = run_experiment_suite(exp_key, TRAIN_CONFIG, verbose=True)
        all_results[exp_key] = results
        plot_experiment_results(results, exp_key)
        plot_convergence_check(results, exp_key)

    # 摘要
    plot_summary_table(all_results)
    save_results_json(all_results)

    # 如果跑了多個實驗，畫綜合比較圖
    if len(all_results) > 1:
        plot_all_experiments_comparison(all_results)

    total_elapsed = time.time() - total_start
    print(f"\n總計耗時: {total_elapsed / 60:.1f} 分鐘")
    print("完成！請查看 results/ 資料夾。")
    print("\n提示:")
    print("  1. 先確認收斂曲線 (convergence_*.png) 是否穩定")
    print("  2. 再看敏感度圖 (sensitivity_*.png) 的趨勢")
    print("  3. 若要跑全部: experiments_to_run = ['theta', 'hf', 's', 'c']")
    print("  4. 若要多次重複取均值: 修改 TRAIN_CONFIG['num_seeds'] = 3")