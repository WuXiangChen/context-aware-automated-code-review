from copy import deepcopy
import copy
import hashlib
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from Database.evaluationCache import EvaluationCache
from Model.actorCritic.collectEpisodesMerged import collect_episodes_merged
from Model import *
from AllocMethod import *
from Model.actorCritic.trainingVisualizer import TrainingVisualizer
from Model.actorCritic.util import pad_trajectories_to_numpy
from Utils.rl_util import evaluate_batch_examples, generate_context_from_nodeList, get_simple_model_signature, preprocess_graph_data
from typing import Optional
from tqdm import tqdm
logger = logging.getLogger("Output/logs/"+__name__)
# state定义为什么？很简单，就是已选节点集合的context embedding
# critic的输入模型是什么？很简单，就是context embedding，codediff embedding 和 对应某个state的action；这表明模型在学习的是当前(s,a)状态-动作对的Q函数
# actor的输入模型是什么？ 很简单，就是当前图的节点特征（图上每个节点的节点特征，就定义为其自然文本拼接的embedding值）以及边集合。
class SimplifiedRLTrainer:
    """简化的强化学习训练器 - 配合预设的Actor-Critic架构"""
    def __init__(self, args, actor, critic, predefined_bleu_model, rfDataTrans, pool, predefined_args):
        self.args = args
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device(predefined_args.device)
        self.device = torch.device(predefined_args.device)
        print(f"device:{torch.device(predefined_args.device)}")
        
        # 使用传入的Actor和Critic网络
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.predefined_bleu_model = predefined_bleu_model.to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # 训练统计
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        self.rfDataTrans = rfDataTrans
        self.pool = pool
        self.predefined_args = predefined_args
        
        if self.predefined_args.task_type == "msg":
            self.cached_file_name = f"evaluation_cache_msg_codediff_{predefined_args.maxRL}.json"
        elif self.predefined_args.task_type == "ref":
            self.cached_file_name = "evaluation_cache_ref.json"
        self.ec = EvaluationCache(self.cached_file_name)
        self.SAVE_COUNTER = 0
    
    def flush_ec(self):
        self.ec.save_cache()
        self.ec = EvaluationCache(self.cached_file_name)
        self.SAVE_COUNTER = 0
        
    def get_sha256(self, crItem):
        """
        计算最终奖励的成员方法
        
        Args:
            current_nodes: 当前节点列表
            idx_to_node: 索引到节点的映射
            graph_data: 图数据
            crItem: 当前项目字典
            predefined_args: 预定义参数
        
        Returns:
            final_reward: 计算得到的最终奖励值
        """
        # TODO 3: 实现这里将embedding作为输入，并对应生成文本输出与bleu reward的方法
        model_info = ""
        if hasattr(self.predefined_bleu_model, 'config') and hasattr(self.predefined_bleu_model.config, 'vocab_size'):
            model_info += get_simple_model_signature(
                self.predefined_bleu_model, 
                beam_size=self.predefined_args.beam_size, 
                tokenizer=self.rfDataTrans.tokenizer, 
                args=self.predefined_args
            )

        # 生成缓存键
        combined_input = f"{crItem['context']}###MODEL###{model_info}"
        sha_context = hashlib.sha256(combined_input.encode()).hexdigest()
        
        return sha_context

    def calculate_bleu_from_cache(self, crItem):
        """
        从缓存中获取BLEU分数
            
        Returns:
            tuple: (是否命中缓存, BLEU分数) 
                如果命中返回 (True, score)
                如果未命中返回 (False, None)
        """
        sha_context = self.get_sha256(crItem)
        
        if sha_context in self.ec.cache:
            return self.ec.cache[sha_context]
        else:
            # print(f"Cache miss for SHA: {sha_context[:8]}...")
            return False

    def calculate_bleu_from_preTrainedModel(self, states, crItems, saved_flag=True):
        """
        使用预训练模型计算BLEU分数
        
        Args:
            state: 当前状态
            combined_input: 用于生成缓存键的输入字符串
            
        Returns:
            float: 计算得到的BLEU分数
        """
        # 批量生成缓存键
        sha_contexts = [self.get_sha256(crItem) for crItem in crItems]
        # 执行评估
        final_bleu, all_other_metrics = evaluate_batch_examples(
            model=self.predefined_bleu_model, 
            examples=states, 
            device=self.device, 
            beam_size=self.predefined_args.beam_size, 
            tokenizer=self.rfDataTrans.tokenizer,
            llm_device=self.predefined_args.llm_device
        )
        
        # 存储到缓存
        if saved_flag:
            for k, sha_context in enumerate(sha_contexts):
                bleu = final_bleu[k]
                all_metrics = all_other_metrics[k]
                all_metrics["bleu"] = bleu
                self.ec.cache[sha_context] = all_metrics
                self._increment_save_counter()
            print(f"💾 New evaluation: {final_bleu}")
            torch.cuda.empty_cache()
        return final_bleu, all_other_metrics

    def _increment_save_counter(self):
        """
        增加保存计数器并在必要时刷新缓存
        """
        self.SAVE_COUNTER += 1
        if self.SAVE_COUNTER > 2000:  # 每100次保存
            self.flush_ec()
    
    def compute_advantages_for_episodes(self, trajectories, rewards, q_values):
        """为episode轨迹计算advantages"""
        returns = []
        advantages = []
        
        # 按episode分组处理
        episode_starts = [0]
        for i, step in enumerate(trajectories):
            if step['is_terminal']:
                episode_starts.append(i + 1)
        
        for start_idx in range(len(episode_starts) - 1):
            episode_start = episode_starts[start_idx]
            episode_end = episode_starts[start_idx + 1]
            
            # 计算这个episode的returns
            episode_rewards = rewards[episode_start:episode_end]
            episode_q_values = q_values[episode_start:episode_end]
            
            # 从后往前计算discounted returns
            episode_returns = []
            running_return = 0.0
            
            for r in reversed(episode_rewards):
                running_return = r + self.args.gamma * running_return
                episode_returns.append(running_return)
            
            episode_returns.reverse()
            
            # 计算advantages
            episode_advantages = [ret - q_val for ret, q_val in zip(episode_returns, episode_q_values)]
            
            returns.extend(episode_returns)
            advantages.extend(episode_advantages)
        
        return torch.tensor(returns, device=self.device), torch.tensor(advantages, device=self.device)

    def collect_episode(self, crItems, batch_idx=0, test_flag="train", merged_data=None):
        """收集一个完整的episode"""
        if merged_data is not None:
            merged_data = merged_data
        else:
            core_ids = ["PR-"+str(crItem["ghid"]) for crItem in crItems]
            merged_data = collect_episodes_merged(crItems, batch_idx,  self.rfDataTrans, self.predefined_args, self.device, core_ids, test_flag=test_flag)
            
        batch_data, global_node_to_idx = merged_data["batch_data"], merged_data["global_node_to_idx"]
        crItems, core_ids, node_neighbors = merged_data["crItems"], merged_data["core_ids"], merged_data["node_neighbors"]
        idx_to_global_node =  {v: k for k, v in global_node_to_idx.items()}

        current_nodes=[[index] for index in core_ids]
        num_graphs = len(core_ids)
        pre_bleu = [0.0]*num_graphs
        # 2. 处理每个图的动作和奖励
        trajectories = [[] for _ in range(num_graphs)]
        is_stop = [False] * num_graphs
        index = 0
        max_bleu_per_trajectry = []
        max_other_metrics_per_trajectry = []
        while not all(is_stop):
            pre_bleu = copy.deepcopy(pre_bleu)
            rewards = [None] * len(is_stop)
            states = [None] * len(is_stop)  # 保存每个图的状态
            other_metrics = [{}]*num_graphs
            # 2. 更新每图的 current_nodes 和 is_stop
            batch_data = batch_data.to(self.device)
            action, log_prob, entropy, selected_node, is_stop = self.actor.sample_action(batch_data, current_nodes, node_neighbors, maxRL=self.predefined_args.maxRL)
            for graph_idx in range(num_graphs):
                selected_node_idx = selected_node[graph_idx]
                current_nodes[graph_idx].append(selected_node_idx)
                if is_stop[graph_idx] and index!=0:
                    # 如果该图已经停止，奖励增益为0
                    reward = 0.000001
                else:
                    metrics = self.calculate_bleu_from_cache(crItems[graph_idx])
                    if metrics != False:  
                        cur_bleu = metrics["bleu"]
                        reward = cur_bleu - pre_bleu[graph_idx] + 0.000001
                        pre_bleu[graph_idx] = cur_bleu
                        other_metrics[graph_idx] = metrics
                    else:
                        reward = False
                if self.predefined_args.task_type == "msg":
                    state = self.rfDataTrans.tokenize(item=(crItems[graph_idx], self.rfDataTrans.tokenizer, self.predefined_args))
                elif self.predefined_args.task_type == "ref":
                    state = self.rfDataTrans.tokenize_for_ref(item=(crItems[graph_idx], self.rfDataTrans.tokenizer, self.predefined_args))
                else:
                    raise "Only msg and ref could be assigned to task_type in args"
                rewards[graph_idx] = reward
                states[graph_idx] = state
            if not all(rewards):
                incomplete_indices = [i for i, status in enumerate(rewards) if status==False]
                # 提取对应的数据和状态
                reeval_states = [states[i] for i in incomplete_indices]
                reeval_crItems = [crItems[i] for i in incomplete_indices]
                completed_bleu, ep_other_metrics = self.calculate_bleu_from_preTrainedModel(reeval_states, reeval_crItems)
                for i, missed_index in enumerate(incomplete_indices):
                    rewards[missed_index] = completed_bleu[i] - pre_bleu[missed_index]
                    pre_bleu[missed_index] = completed_bleu[i]
                    other_metrics[missed_index] = ep_other_metrics[i]
                
            # 3. 记录轨迹（为每个图记录）
            for graph_idx in range(num_graphs):
                # 将current_nodes转换为全局节点ID
                node_ids = [idx_to_global_node[id_].split("node_")[-1] for id_ in current_nodes[graph_idx] if id_ in idx_to_global_node.keys()]
                trajectories[graph_idx].append({
                    'state': torch.tensor(states[graph_idx].source_ids, dtype=torch.float32),
                    'action': selected_node[graph_idx],
                    'log_prob': log_prob[graph_idx].to(self.device),
                    'entropy': entropy[graph_idx].to(self.device),
                    'reward': rewards[graph_idx],
                    'is_terminal': is_stop[graph_idx],
                    'node_ids': node_ids
                })
                # 更新该图的context
                crItems[graph_idx]["context"] = generate_context_from_nodeList(crItems[graph_idx]["graph_data"], node_ids)
            
            max_bleu_per_trajectry.append(pre_bleu)
            max_other_metrics_per_trajectry.append(other_metrics)
            # 清显卡
            torch.cuda.empty_cache()
            index+=1
            # print(f"index:{index}")
        all_repo_ids = [item["repo_id"] for item in crItems]
        return trajectories, max_bleu_per_trajectry, max_other_metrics_per_trajectry, all_repo_ids
    
    def evaluate(self, dataloader, test_flag):
        """评估模型（多图多step版本）"""
        self.actor.eval()
        self.critic.eval()
        episode_other_metrics = []
        all_rewards = []  # 每个episode的总reward
        all_repo_ids = []
        all_selected_nodes = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # 跑十遍收集episode，找最大值
                all_max_bleu_per_trajectory = []
                all_other_metrics = []
                all_trajactories = []
                core_ids = ["PR-"+str(crItem["ghid"]) for crItem in batch_data]
                merged_data = collect_episodes_merged(batch_data, batch_idx,  self.rfDataTrans, self.predefined_args, self.device, core_ids, test_flag=test_flag)
                
                for run_idx in range(20):
                    # 收集每个图的episode
                    # print("run_idx:run_idx")
                    print("run_idx:{run_idx}")
                    trajactory, max_bleu_per_trajectory, other_metrics, episode_repo_ids = self.collect_episode(batch_data, batch_idx, test_flag=test_flag, merged_data=merged_data)
                    all_max_bleu_per_trajectory.extend(max_bleu_per_trajectory)
                    all_other_metrics.extend(other_metrics)
                    all_trajactories.extend(trajactory)
                    
                    # 将10次运行的结果合并
                    np_max_bleu = pad_trajectories_to_numpy(all_max_bleu_per_trajectory)
                    max_indices = np.argmax(np_max_bleu, axis=0)  # 找每列的最大值索引
                    # 选择对应的最佳结果
                    selected_other_metrics = []
                    selected_context_nodeList = []
                    for trajectory_idx in range(len(max_indices)):
                        best_run_idx = max_indices[trajectory_idx]
                        selected_other_metrics.append(all_other_metrics[best_run_idx][trajectory_idx])
                        selected_context_nodeList.append(all_trajactories[trajectory_idx][best_run_idx%(self.predefined_args.maxRL+1)]["node_ids"])
                    
                    episode_other_metrics.extend(selected_other_metrics)
                    col_maxes = np.max(np_max_bleu, axis=0)  # 每列的最大值
                    all_rewards.extend(col_maxes)
                    all_repo_ids.extend(episode_repo_ids)  # 注意：这里可能需要调整，因为现在有10次运行
                    all_selected_nodes.extend(selected_context_nodeList)
                
                    mean_reward = np.mean(all_rewards) if all_rewards else 0.0
                    print(f"run_idx:{run_idx}, mean_bleu:{mean_reward}")
        df = None
        if test_flag=="test":
            df = pd.DataFrame(episode_other_metrics)
            other_metrics_df = df.mean().to_dict()
            df['bleu'] = all_rewards
            df['repo_id'] = all_repo_ids
            df['selected_nodeLs'] = all_selected_nodes
            other_metrics_df["bleu"] = mean_reward
        self.actor.train()
        self.critic.train()
        return mean_reward, df

    def train_step(self, batch_data, batch_idx):
        """单步训练 - 处理完整episodes"""
        # batch_data 就是一个list
        trajectories_start_time = time.time()
        # 1. 收集所有episodes的轨迹
        all_trajectories, _, _, _ = self.collect_episode(batch_data, batch_idx)
        trajectories_end_time = time.time()
        logger.info(f"长度为{len(batch_data)}的一个批次，其生成该批次的全部trajectories的时长为{trajectories_end_time-trajectories_start_time}秒")
        
        # all_trajectories: List[List[Dict]]，每个元素是一个图的episode（多step）
        # --- collect flattened lists and predicted values ---
        states = []
        actions = []
        log_probs = []
        rewards = []
        entropies = []
        is_terminals = []
        predicted_state_values = []
        state_values = []
        
        # 第一步：收集所有数据
        all_step_states = []
        all_step_count = 0
        for episode in all_trajectories:
            for step in episode:
                states.append(step['state'])
                actions.append(step.get('action'))
                log_probs.append(step['log_prob'])
                rewards.append(float(step['reward']))
                entropies.append(step['entropy'])
                is_terminals.append(bool(step['is_terminal']))
                all_step_states.append(step['state'])
                if is_terminals!=True:
                    all_step_count+=1
        
        # 第二步：批量计算critic values (这是唯一的优化部分)
        if all_step_states:
            batch_states = torch.stack(all_step_states).to(self.device)  # [N, 256]
            batch_v_vals = self.critic(batch_states).view(-1)            # [N]
            
            # 转换回原来的格式
            for v_val in batch_v_vals:
                predicted_state_values.append(v_val.view(1))
                state_values.append(v_val.item())

        # stack tensors
        log_probs = torch.stack(log_probs)                # [N]
        entropies = torch.stack(entropies)                # [N]
        predicted_state_values = torch.cat(predicted_state_values, dim=0).to(self.device).float()  # [N]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)            # [N]
        is_terminals = np.array(is_terminals, dtype=np.bool_)                                        # [N] (numpy for indexing)

        # --- GAE parameters ---
        gamma = getattr(self.args, 'gamma', 0.99)
        lam = getattr(self.args, 'gae_lambda', 0.95)

        # --- compute GAE per-episode (reverse) ---
        advantages_list = []
        returns_list = []
        idx = 0
        for episode in all_trajectories:
            ep_len = len(episode)
            if ep_len == 0:
                continue

            # slice episode tensors
            ep_rewards = rewards_tensor[idx: idx + ep_len]               # tensor on device
            ep_values = predicted_state_values[idx: idx + ep_len]        # tensor on device
            ep_terminal = is_terminals[idx: idx + ep_len]                # numpy bool array

            # prepare advantage tensor
            advantages = torch.zeros(ep_len, device=self.device)
            lastgaelam = 0.0

            # reversed GAE
            for t in range(ep_len - 1, -1, -1):
                # next_value = v_{t+1} if exists and not terminal at t (i.e., if this step is not terminal),
                # else 0. We treat ep_terminal[t] == True meaning this step is terminal (no bootstrap).
                if t == ep_len - 1:
                    next_value = 0.0
                    next_nonterminal = 0.0 if ep_terminal[t] else 1.0
                else:
                    # next_value = value[t+1] (bootstrap)
                    next_value = ep_values[t + 1].item()
                    next_nonterminal = 0.0 if ep_terminal[t] else 1.0

                delta = ep_rewards[t].item() + gamma * next_value * next_nonterminal - ep_values[t].item()
                lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam

            # returns = advantages + values
            ep_returns = advantages + ep_values.detach()
            advantages_list.append(advantages)
            returns_list.append(ep_returns)

            idx += ep_len

        # concat episode results to single tensors (same order as flattened lists)
        if len(advantages_list) > 0:
            advantages = torch.cat(advantages_list, dim=0)   # [N]
            returns_tensor = torch.cat(returns_list, dim=0)  # [N]
        else:
            # fallback to zeros if no data
            advantages = torch.zeros(0, device=self.device)
            returns_tensor = torch.zeros(0, device=self.device)

        # --- advantage normalization (recommended) ---
        if advantages.numel() > 0:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            # keep a copy for logging before detach if needed
        else:
            advantages = advantages

        # clamp advantages optionally (kept small clipping as safety)
        advantages = torch.clamp(advantages, -10, 10)

        # --- critic loss (fit V(s) to returns) ---
        # predicted_state_values is V(s) for each step
        critic_loss = F.smooth_l1_loss(predicted_state_values, returns_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.critic_optimizer.step()

        # --- actor loss (policy gradient) ---
        # note: minimize loss -> we put negative sign for policy gradient and subtract entropy bonus
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = entropies.mean()  # want to maximize entropy -> subtract in loss
        actor_loss = policy_loss - self.args.entropy_weight * entropy_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()

        # --- logging / returns ---
        total_steps = all_step_count
        avg_reward = float(torch.mean(torch.tensor([s['reward'] for ep in all_trajectories for s in ep], device=self.device))) if total_steps > 0 else 0.0
        avg_advantage = float(advantages.mean().item()) if advantages.numel() > 0 else 0.0
        trajectory_sums = [sum(s['reward'] for s in ep) for ep in all_trajectories]
        avg_sum_reward = float(torch.mean(torch.tensor(trajectory_sums, device=self.device)))
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_reward': avg_reward,
            'avg_advantage': avg_advantage,
            'avg_sum_reward': avg_sum_reward,
            'avg_episode_length': total_steps / max(1, len(all_trajectories))
        }

    def train_epoch(self, dataloader, epoch_num=None, visualizer: Optional['TrainingVisualizer'] = None):
        """训练一个epoch（支持多图多step的batch）- 集成可视化功能"""
        self.actor.train()
        self.critic.train()
        
        epoch_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'rewards': [],
            'advantages': [],
            'episode_lengths': [],
            'avg_sum_reward': []
        }
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_start_time = time.time()
            # # 训练一个batch（多图多step）
            batch_idx += self.predefined_args.baseBatchIndex
            stats = self.train_step(batch_data, batch_idx)
            
            # 累积统计
            epoch_stats['actor_losses'].append(stats['actor_loss'])
            epoch_stats['critic_losses'].append(stats['critic_loss'])
            epoch_stats['rewards'].append(stats['avg_reward'])
            epoch_stats['advantages'].append(stats['avg_advantage'])
            epoch_stats['episode_lengths'].append(stats['avg_episode_length'])
            epoch_stats['avg_sum_reward'].append(stats["avg_sum_reward"])
            
            batch_time = time.time() - batch_start_time
            
            # 可视化记录
            if visualizer:
                batch_metrics = {
                    'actor_loss': stats['actor_loss'],
                    'critic_loss': stats['critic_loss'],
                    'avg_reward': stats['avg_reward'],
                    'avg_advantage': stats['avg_advantage'],
                    'avg_episode_length': stats['avg_episode_length'],
                    'avg_sum_reward':np.mean(epoch_stats['avg_sum_reward']),
                    'batch_time': batch_time
                }
                visualizer.log_batch_metrics(batch_metrics, batch_idx)
                
                if batch_idx % self.args.log_interval == 0:
                    visualizer.log_learning_rates({
                        'Actor': self.actor_optimizer,
                        'Critic': self.critic_optimizer
                    })
        # Epoch统计
        avg_stats = {
            'avg_actor_loss': np.mean(epoch_stats['actor_losses']),
            'avg_critic_loss': np.mean(epoch_stats['critic_losses']),
            'avg_reward': np.mean(epoch_stats['rewards']),
            'all_return': np.sum(epoch_stats['rewards']),
            'avg_advantage': np.mean(epoch_stats['advantages']),
            'avg_episode_length': np.mean(epoch_stats['episode_lengths']),
            'std_reward': np.std(epoch_stats['rewards']),
            'avg_sum_reward':np.mean(epoch_stats['avg_sum_reward']),
            'epoch_time': time.time() - start_time
        }
        
        # 可视化Epoch总结
        if visualizer:
            visualizer.log_epoch_summary(epoch_num or 0, avg_stats)
        
        return avg_stats

    # 规划train与test的部分
    def run(self, train_dataloader, val_dataloader=None, num_epochs=1000):
        """运行训练（多图多step）"""
        logger.info("Starting Simplified RL Training")
        
        best_reward = -float('inf')
        visualizer = TrainingVisualizer(experiment_name=self.predefined_args.rlModel)
        for epoch in range(num_epochs):
            # 训练
            train_stats = self.train_epoch(train_dataloader,  epoch_num=epoch, visualizer=visualizer)
            # 一个epoch就是一轮训练（多个图）
            self.episode_rewards.append(train_stats['avg_reward'])
            self.actor_losses.append(train_stats['avg_actor_loss'])
            self.critic_losses.append(train_stats['avg_critic_loss'])
            
            if ((epoch % 10 == 0 and epoch!=0 and epoch < 80) or (epoch % 1 == 0 and epoch > 80)) and val_dataloader is not None:
                val_reward, _ = self.evaluate(val_dataloader, test_flag="valid")
                
                # 记录到 TensorBoard
                re = {
                    "Validation/Reward": val_reward,
                    "Train/Reward": train_stats['avg_reward'],
                    "Train/Actor_Loss": train_stats['avg_actor_loss'],
                    "Train/Critic_Loss": train_stats['avg_critic_loss']
                }
                visualizer.log_scalar("Validation/Reward", val_reward, step=epoch)
                visualizer.log_scalars({
                    "Validation/Reward": val_reward,
                    "Train/Reward": train_stats['avg_reward'],
                    "Train/Actor_Loss": train_stats['avg_actor_loss'],
                    "Train/Critic_Loss": train_stats['avg_critic_loss']
                }, step=epoch)
                print(re)
                # 保存最佳模型
                if val_reward > best_reward:
                    best_reward = val_reward
                    self.save_models(epoch, is_best=True)
                else:
                    self.save_models(epoch, is_best=False)

    def save_models(self, epoch, is_best=False):
        """保存模型"""
        suffix = '_best' if is_best else f'_epoch_{epoch}'
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epoch': epoch,
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, f'{self.args.output_dir}/rl_model{suffix}.pt')
        logger.info(f"Model saved: rl_model{suffix}.pt")

    def load_models(self, model_path, load_optimizer=True):
        """加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 加载模型参数
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            # 可选择是否加载优化器状态
            if load_optimizer:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # 加载训练历史信息
            epoch = checkpoint.get('epoch', 0)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.actor_losses = checkpoint.get('actor_losses', [])
            self.critic_losses = checkpoint.get('critic_losses', [])
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Resumed from epoch {epoch}")
            
            return epoch
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except KeyError as e:
            logger.error(f"Missing key in checkpoint: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise