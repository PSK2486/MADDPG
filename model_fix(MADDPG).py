import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import copy

# ==============================================================================
# 0. Device Configuration
# ==============================================================================
device = torch.device( "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. 論文模型參數 (Economic Environment Parameters)
# ==============================================================================
# 參考論文參數設定 
PARAM_V = 170.0   # 產品保留價值 v
PARAM_S = 25.0    # 原始運費 s (未補貼前)
PARAM_T = 15.0    # 線上等待成本 t 
PARAM_MC = 30.0   # 邊際成本
PARAM_C = 2.0     # BOPS 運營成本 c 
FIXED_COST_BOPS = 5.0 # 固定成本

# 1. 增加品牌偏好係數，防止價格戰導致一家倒閉
PARAM_T_HOTELLING = 20.0 

# --- Hotelling 模型參數 ---
# 論文 : unit misfit cost = 1 (所以不用額外乘係數)
# 論文 : theta 是 misfit cost 的折扣因子 (0 < theta < 1)
PARAM_THETA = 0.8  

# --- 消費者分群參數 ---
# Store-only (beta) vs Omnichannel (1-beta)
PARAM_BETA = 0.3  
# Low travel cost (alpha) vs High travel cost (1-alpha)
PARAM_ALPHA = 0.5 

# --- 移動/拜訪成本 (Travel Cost h_f) ---
# h_L = 0 (或很小), h_H 很大
PARAM_H_L = 2.0   
PARAM_H_H = 50.0  # 設大一點確保他們偏好線上 

# 動作邊界
MAX_PRICE = 170.0 
NUM_CONSUMERS = 1000

# ==============================================================================
# 2. 超參數 (Hyperparameters) - 調整以促進收斂
# ==============================================================================
learning_rate_actor = 0.0001   # 提高 Actor 學習率，讓它能跟上 Critic
learning_rate_critic = 0.001   # 提高 Critic 學習率
discount_factor = 0.95
tau = 0.005                    # 稍微加快軟更新速度
num_episodes = 6000          
batch_size = 256               # 增加 Batch Size 以穩定梯度
buffer_size = 200000
max_steps_per_episode = 10     # 縮短每回合步數，因為這是靜態賽局，不需要太長
exploration_noise_start = 0.5
noise_decay = 0.9992           # 加快衰減：約在 5000 回合降至最低
min_exploration_noise = 0.01

NUM_AGENTS = 2
ACTION_SIZE = 3 # (Price, BOPS, Phi)
GLOBAL_STATE_SIZE = ACTION_SIZE * NUM_AGENTS 
TOTAL_ACTION_SIZE = ACTION_SIZE * NUM_AGENTS
ACTION_MAX_VECTOR = torch.tensor([MAX_PRICE, 1.0, 1.0], dtype=torch.float32).to(device)

# ==============================================================================
# 3. 網路定義
# ==============================================================================
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_max_vector):
        super(ActorNetwork, self).__init__()
        self.action_max_vector = action_max_vector
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = torch.sigmoid(self.fc3(x)) 
        return action

class CriticNetwork(nn.Module):
    def __init__(self, state_size, total_action_size):
        super(CriticNetwork, self).__init__()
        self.fc_s = nn.Linear(state_size, 64)
        self.fc_a = nn.Linear(total_action_size, 64)
        self.fc_cat = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, state, action):
        s = F.relu(self.fc_s(state))
        a = F.relu(self.fc_a(action))
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc_cat(cat))
        q_value = self.fc_out(x)
        return q_value

# ==============================================================================
# 4. Agent
# ==============================================================================
class Agent:
    def __init__(self, agent_id, state_size, action_size, total_action_size, action_max_vector):
        self.agent_id = agent_id
        self.action_size = action_size
        self.action_max = action_max_vector
        self.actor = ActorNetwork(state_size, action_size, action_max_vector).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic = CriticNetwork(state_size, total_action_size).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, noise_std_dev):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            raw_action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        noise = np.random.normal(0, noise_std_dev, size=self.action_size)
        noisy_action = np.clip(raw_action + noise, 0.0, 1.0)
        return noisy_action

def scale_action(action):
    """ 將 [0,1] 縮放回真實環境數值 """
    scaled = np.copy(action)
    min_price = PARAM_MC + 5.0
    scaled[0] = min_price + action[0] * (MAX_PRICE - min_price)
    return scaled

def scale_action_torch(action_tensor):
    min_price = PARAM_MC + 5.0
    scaled = action_tensor.clone()
    for i in range(scaled.shape[1] // ACTION_SIZE):
        idx = i * ACTION_SIZE
        scaled[:, idx : idx+1] = min_price + action_tensor[:, idx : idx+1] * (MAX_PRICE - min_price)
    return scaled

# ==============================================================================
# 5. Replay Buffer
# ==============================================================================
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
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
class MADDPG:
    def __init__(self, num_agents, state_size, action_size, total_action_size, action_max_vector):
        self.num_agents = num_agents
        self.agents = [Agent(i, state_size, action_size, total_action_size, action_max_vector) for i in range(num_agents)]
        self.buffer = ReplayBuffer(buffer_size)
        self.train_step_count = 0

    def get_actions(self, state, noise_std_dev):
        actions = [agent.select_action(state, noise_std_dev) for agent in self.agents]
        return np.array(actions)

    def store_experience(self, state, actions, rewards, next_state):
        self.buffer.add(state, actions, rewards, next_state)

    def train(self, batch_size, discount_factor, tau):
        if len(self.buffer) < batch_size: return
        self.train_step_count += 1 
        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        states_tensor = torch.FloatTensor(states).to(device)
        
        # 縮放動作給 Critic
        raw_actions_tensor = torch.FloatTensor(actions).view(batch_size, -1).to(device)
        actions_tensor = scale_action_torch(raw_actions_tensor)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)

        target_next_actions_all = []
        with torch.no_grad():
            for agent in self.agents:
                pred = agent.target_actor(next_states_tensor)
                target_next_actions_all.append(scale_action_torch(pred))
        target_next_actions_all_tensor = torch.cat(target_next_actions_all, dim=1)

        for i, agent in enumerate(self.agents):
            agent_rewards = rewards_tensor[:, i].unsqueeze(1)
            with torch.no_grad():
                target_q = agent.target_critic(next_states_tensor, target_next_actions_all_tensor)
                targets = agent_rewards + discount_factor * target_q
            current_q = agent.critic(states_tensor, actions_tensor)
            critic_loss = F.mse_loss(current_q, targets)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            # 延遲更新 Actor
            if self.train_step_count % 2 == 0:
                actions_pred_all = []
                for j, other_agent in enumerate(self.agents):
                    pred = other_agent.actor(states_tensor)
                    actions_pred_all.append(scale_action_torch(pred)) 
                actions_pred_all_tensor = torch.cat(actions_pred_all, dim=1)
                
                actor_loss = -agent.critic(states_tensor, actions_pred_all_tensor).mean()
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor_optimizer.step()

        if self.train_step_count % 2 == 0:
            for agent in self.agents:
                agent.update_target_networks(tau)

# ==============================================================================
# 7. 核心環境函數: 計算利潤 (已修正為 Hotelling 模型)
# ==============================================================================
def calculate_profits_numerical(action1, action2):
    p1, bops_val1, phi1 = action1
    p2, bops_val2, phi2 = action2
    
    # Sigmoid 平滑化權重 (for cost calculation)
    k = 15.0  
    w1 = 1.0 / (1.0 + np.exp(-k * (bops_val1 - 0.5)))
    w2 = 1.0 / (1.0 + np.exp(-k * (bops_val2 - 0.5)))
    
    # [核心修正] x 代表消費者在 Hotelling 線上的位置 [0, 1]
    # x=0 代表 Retailer 1 的位置，x=1 代表 Retailer 2 的位置
    x_loc = np.linspace(0, 1, NUM_CONSUMERS) 
    dist1 = x_loc        # 到 R1 的距離
    dist2 = 1 - x_loc    # 到 R2 的距離

    def get_segment_choices(h_cost, force_store_only=False):
        # 使用 PARAM_T_HOTELLING 增加市場摩擦力
        u_s1 = PARAM_V - p1 - PARAM_T_HOTELLING * dist1 - h_cost
        u_s2 = PARAM_V - p2 - PARAM_T_HOTELLING * dist2 - h_cost

        if force_store_only:
            u_o1 = u_o2 = u_b1 = u_b2 = np.full_like(x_loc, -9999.0)
        else:
            # 2. 線上效用 (Online)
            # 論文 Eq(3) & (4) : U_O = v - p - theta*x - s - t
            # s: 消費者負擔的運費 = (1-phi)*S 
            s1_consumer = (1 - phi1) * PARAM_S
            s2_consumer = (1 - phi2) * PARAM_S
            
            u_o1 = PARAM_V - p1 - PARAM_THETA * PARAM_T_HOTELLING * dist1 - s1_consumer - PARAM_T
            u_o2 = PARAM_V - p2 - PARAM_THETA * PARAM_T_HOTELLING * dist2 - s2_consumer - PARAM_T
            
            # 3. BOPS 效用 (BOPS)
            # 論文 Eq(5) & (6) : U_B = v - p - theta*x - h_f
            # 論文 指出 BOPS 的 misfit 也是 theta*x
            
            # 若沒開 BOPS，給予懲罰
            bops_penalty1 = (1.0 - w1) * 200.0
            bops_penalty2 = (1.0 - w2) * 200.0
            
            u_b1 = PARAM_V - p1 - PARAM_THETA * PARAM_T_HOTELLING * dist1 - h_cost - bops_penalty1
            u_b2 = PARAM_V - p2 - PARAM_THETA * PARAM_T_HOTELLING * dist2 - h_cost - bops_penalty2

        u_no_buy = np.zeros_like(x_loc)
        all_utilities = np.vstack([u_s1, u_o1, u_b1, u_s2, u_o2, u_b2, u_no_buy])
        choices = np.argmax(all_utilities, axis=0)
        
        counts = np.zeros(7)
        for i in range(7):
            counts[i] = np.sum(choices == i) / NUM_CONSUMERS
        return counts[:6]

    # 計算分群需求
    d_beta = get_segment_choices(PARAM_H_L, force_store_only=True)
    d_alpha_low = get_segment_choices(PARAM_H_L, force_store_only=False)
    d_alpha_high = get_segment_choices(PARAM_H_H, force_store_only=False) # High h_f -> Prefer Online

    total_demand_dist = PARAM_BETA * d_beta + \
                        (1 - PARAM_BETA) * (PARAM_ALPHA * d_alpha_low + (1 - PARAM_ALPHA) * d_alpha_high)

    d_s1, d_o1, d_b1 = total_demand_dist[0], total_demand_dist[1], total_demand_dist[2]
    d_s2, d_o2, d_b2 = total_demand_dist[3], total_demand_dist[4], total_demand_dist[5]

    # 利潤計算 (廠商付出的補貼是 (1-phi)*S 的補集? 不，運費補貼是廠商吸收的部分)
    # 這裡邏輯：消費者付 (1-phi)S。
    # 廠商收入：P - MC。
    # 廠商成本：補貼運費 phi*S (若 phi=1, 廠商全付; phi=0, 消費者全付)
    profit1 = (p1 - PARAM_MC) * (d_s1 + d_o1 + d_b1) - (phi1) * PARAM_S * d_o1 - PARAM_C * d_b1
    profit2 = (p2 - PARAM_MC) * (d_s2 + d_o2 + d_b2) - (phi2) * PARAM_S * d_o2 - PARAM_C * d_b2
    
    profit1 -= w1 * FIXED_COST_BOPS
    profit2 -= w2 * FIXED_COST_BOPS

    return profit1, profit2, [d_s1, d_o1, d_b1], [d_s2, d_o2, d_b2]

# ==============================================================================
# 8. 主訓練迴圈 - 修正 Reward Shaping 與 邏輯
# ==============================================================================
if __name__ == "__main__":
    maddpg = MADDPG(NUM_AGENTS, GLOBAL_STATE_SIZE, ACTION_SIZE, TOTAL_ACTION_SIZE, ACTION_MAX_VECTOR)
    
    history_rewards = []
    current_noise = exploration_noise_start
    current_state = np.zeros(GLOBAL_STATE_SIZE)

    print(f"--- 開始訓練 MADDPG (Hotelling Model Corrected) ---")
    best_combined_profit = -float('inf')
    
    for episode in range(num_episodes):
        step_rewards = np.zeros(NUM_AGENTS)
        # 每一回合開始時，重置參考動作為當前 Actor 的預測（不含雜訊）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)
            initial_raw_actions = [agent.actor(state_tensor).cpu().numpy()[0] for agent in maddpg.agents]
            prev_env_actions = [scale_action(a) for a in initial_raw_actions]
        
        prev_bops_state = np.array([1 if a[1] > 0.5 else 0 for a in initial_raw_actions])
        
        for step in range(max_steps_per_episode):
            raw_actions = maddpg.get_actions(current_state, current_noise)
            env_actions = [scale_action(a) for a in raw_actions]
            
            r1, r2, _, _ = calculate_profits_numerical(env_actions[0], env_actions[1])

            # --- 優化版 Reward Shaping ---
            # 1. 降低價格跳動懲罰的敏感度，只針對「劇烈」跳動
            price_jump_penalty_weight = 5.0
            p1_jump = abs(env_actions[0][0] - prev_env_actions[0][0]) / MAX_PRICE
            p2_jump = abs(env_actions[1][0] - prev_env_actions[1][0]) / MAX_PRICE
            
            # 使用平方項，讓小跳動（雜訊引起）不痛，大跳動（策略劇變）重罰
            p1_jump_penalty = (p1_jump ** 2) * price_jump_penalty_weight
            p2_jump_penalty = (p2_jump ** 2) * price_jump_penalty_weight

            reward_scale = 100.0 # 配合 PARAM_T_HOTELLING=40，利潤會變高，調大 scale
            
            reward1 = (r1 / reward_scale) - p1_jump_penalty
            reward2 = (r2 / reward_scale) - p2_jump_penalty
            
            # BOPS 切換懲罰 (僅在確實切換時)
            curr_bops_state = np.array([1 if raw_actions[0][1] > 0.5 else 0, 1 if raw_actions[1][1] > 0.5 else 0])
            if curr_bops_state[0] != prev_bops_state[0]: reward1 -= 0.2
            if curr_bops_state[1] != prev_bops_state[1]: reward2 -= 0.2

            rewards = np.array([reward1, reward2], dtype=np.float32)

            # 更新參考值
            prev_env_actions = env_actions

            next_state = np.concatenate(raw_actions)
            maddpg.store_experience(current_state, raw_actions, rewards, next_state)
            maddpg.train(batch_size, discount_factor, tau)

            current_state = next_state
            step_rewards += rewards
            prev_bops_state = curr_bops_state 

        avg_rewards = step_rewards / max_steps_per_episode
        history_rewards.append(avg_rewards)
        current_noise = max(min_exploration_noise, current_noise * noise_decay)

        if (episode + 1) % 100 == 0:
            ea = env_actions
            print(f"Ep {episode+1} | "
                  f"F1: P={ea[0][0]:.1f}, B={int(ea[0][1]>0.5)}, Phi={ea[0][2]:.2f} | "
                  f"F2: P={ea[1][0]:.1f}, B={int(ea[1][1]>0.5)}, Phi={ea[1][2]:.2f}")

    print("訓練完成")

    # --- 新增繪圖邏輯 ---
    plt.figure(figsize=(10, 5))
    rewards_np = np.array(history_rewards)
    plt.plot(rewards_np[:, 0], label='Firm 1 Reward')
    plt.plot(rewards_np[:, 1], label='Firm 2 Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards over Episodes')
    plt.legend()
    plt.show()