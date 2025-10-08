import hashlib
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from Database.evaluationCache import EvaluationCache
from Model import *
from AllocMethod import *
from Utils.rl_util import evaluate_single_example, generate_context_from_nodeList, get_simple_model_signature, preprocess_graph_data

logger = logging.getLogger("Output/logs/"+__name__)
# state定义为什么？很简单，就是已选节点集合的context embedding
# critic的输入模型是什么？很简单，就是context embedding，codediff embedding 和 对应某个state的action；这表明模型在学习的是当前(s,a)状态-动作对的Q函数
# actor的输入模型是什么？ 很简单，就是当前图的节点特征（图上每个节点的节点特征，就定义为其自然文本拼接的embedding值）以及边集合。
class SimplifiedRLTrainer:
    """简化的强化学习训练器 - 配合预设的Actor-Critic架构"""
    def __init__(self, args, actor, critic, predefined_bleu_model, rfDataTrans, pool, predefined_args):
        self.args = args
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0')
        
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
        
        self.ec = EvaluationCache()
        self.SAVE_COUNTER = 0
        
        self.rfDataTrans = rfDataTrans
        self.pool = pool
        self.predefined_args = predefined_args
    
    def flush_ec(self):
        self.ec.save_cache()
        self.ec = EvaluationCache()
     
    def calculate_final_reward(self, current_nodes, idx_to_node, graph_data, crItem):
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
        # TODO 2: 实现这里自然扩展上下文，并对应生成embedding的方法 ✓
        node_ids = [idx_to_node[item] for item in current_nodes]
        crItem["context"] = generate_context_from_nodeList(graph_data, node_ids)
        state = self.rfDataTrans.tokenize(item=(crItem, self.rfDataTrans.tokenizer, self.predefined_args))
        
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
        
        # 检查缓存
        if sha_context in self.ec.cache:
            print(f"Cache hit for SHA: {sha_context[:8]}...")
            final_reward = self.ec.cache[sha_context]
        else:
            print(f"Cache miss for SHA: {sha_context[:8]}...")
            # 执行评估
            final_reward = evaluate_single_example(
                model=self.predefined_bleu_model, 
                example=state, 
                device=self.device, 
                beam_size=self.predefined_args.beam_size, 
                tokenizer=self.rfDataTrans.tokenizer
            )
            # 存储到缓存
            self.ec.cache[sha_context] = final_reward
            self._increment_save_counter()
            print(f"💾 New evaluation: {final_reward}")
        
        return final_reward

    def _increment_save_counter(self):
        """
        增加保存计数器并在必要时刷新缓存
        """
        self.SAVE_COUNTER += 1
        if self.SAVE_COUNTER % 1000 == 0:  # 每5次保存
            self.flush_ec()
    
    def collect_episode(self, crItem, state):
        """收集一个完整的episode"""
        graph_data = crItem["graph_data"]
        graph_data["node_attributes"]["features"] = []
        node_to_idx = {node: idx for idx, node in enumerate(graph_data["nodes"])}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        for node_id in graph_data["nodes"]:
            node_context = generate_context_from_nodeList(graph_data, [node_id])
            feature = self.rfDataTrans.encode_remove(self.rfDataTrans.tokenizer, node_context, args=self.predefined_args)
            feature, _ = self.rfDataTrans.pad_assert(feature, [], args=self.predefined_args, tokenizer=self.rfDataTrans.tokenizer)
            graph_data["node_attributes"]["features"].append(feature)
                
        node_features, edge_index = preprocess_graph_data(graph_data=graph_data, device=self.device, core_id=core_id, node_to_idx=node_to_idx)
        
        
        core_id = "PR-"+str(crItem["ghid"])
        trajectory = []
        current_nodes=[node_to_idx[core_id]]
        # 改成多图并行的执行
        while True:
            # Actor采样动作，这里采样动作的输入应该是node_features, edge_index, current_nodes，这里需要进行调整
            '''TODO 1: 将这里的sample_action的输入调整正确，最好能搞成标准的输入格式 √'''
            action, log_prob, entropy, selected_node, is_stop = self.actor.sample_action(node_features, edge_index, current_nodes)
            
            if is_stop:
                # 计算最终奖励
                final_reward = self.calculate_final_reward(current_nodes, idx_to_node, graph_data, crItem)
                
                # 记录最后一步
                trajectory.append({
                    'state': torch.tensor(state.source_ids, dtype=torch.float32),
                    'action': action,
                    'log_prob': log_prob,
                    'entropy': entropy,
                    'reward': final_reward,
                    'is_terminal': True
                })
                break
            else:
                # 选择节点，继续episode
                selected_node = action.item()
                current_nodes.append(selected_node)
                # 中间奖励（可以设为0或小的正奖励）
                intermediate_reward = self.calculate_final_reward(current_nodes, idx_to_node, graph_data, crItem)
                trajectory.append({
                    'state': torch.tensor(state.source_ids, dtype=torch.float32),
                    'action': action,
                    'log_prob': log_prob,
                    'entropy': entropy,
                    'reward': intermediate_reward,
                    'is_terminal': False
                })
                
            '''TODO 2: 更新state的变化 √'''
            node_ids = [idx_to_node[item] for item in current_nodes]
            crItem["context"] = generate_context_from_nodeList(graph_data, node_ids)
            state = self.rfDataTrans.tokenize(item=(crItem, self.rfDataTrans.tokenizer, self.predefined_args))
            
        return trajectory
    
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

    def train_step(self, batch_data):
        """单步训练 - 处理完整episodes"""
        # batch_data 就是一个list
        state = self.pool.map(self.rfDataTrans.tokenize, [(example, self.rfDataTrans.tokenizer, self.predefined_args) for example in batch_data])
        batch_size = len(batch_data)
        # 1. 收集所有episodes的轨迹
        all_trajectories = []
        for i in range(batch_size):
            trajectory = self.collect_episode(
                crItem=batch_data[i],
                state=state[i]
            )
            all_trajectories.extend(trajectory)
        
        # 2. 提取轨迹数据
        states = []
        actions = []
        log_probs = []
        rewards = []
        entropies = []
        is_terminals = []
        
        for step in all_trajectories:
            states.append(step['state'])
            actions.append(step['action'])
            log_probs.append(step['log_prob'])
            rewards.append(step['reward'])
            entropies.append(step['entropy'])
            is_terminals.append(step['is_terminal'])
        
        # 转换为tensor
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        entropies = torch.stack(entropies)
        
        # 3&4. 计算V值、returns和advantages
        predicted_state_values = []
        state_values = []

        for i, state in enumerate(states):
            v_val = self.critic(state.to(self.device))  # shape: [1, 1]
            predicted_state_values.append(v_val)
            state_values.append(v_val.item())

        # 拼接为 [batch_size, 1]
        predicted_state_values = torch.cat(predicted_state_values, dim=0).squeeze()

        # 计算 returns（已折扣的累计奖励）和 advantages = returns - V(s)
        returns, _ = self.compute_advantages_for_episodes(all_trajectories, rewards.cpu().numpy(), state_values)
        returns_tensor = returns.clone().detach().to(predicted_state_values.device).float()
        advantages = returns_tensor - predicted_state_values
        advantages = torch.clamp(advantages, -10, 10)
        
        # critic loss：拟合 V(s)
        critic_loss = F.smooth_l1_loss(predicted_state_values, returns_tensor)
        # critic_loss = critic_loss * 0.01
        
        # 5. 更新 critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.critic_optimizer.step()

        # 6. 更新 actor
        actor_loss = -(log_probs * advantages.detach()).mean()
        actor_loss -= self.args.entropy_weight * entropies.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'avg_episode_length': len(all_trajectories) / batch_size
        }
   
    def train_epoch(self, dataloader, epoch_num=None):
        """训练一个epoch"""
        self.actor.train()
        self.critic.train()
        
        epoch_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'rewards': [],
            'advantages': [],
            'episode_lengths': []
        }
        
        total_batches = len(dataloader)
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_start_time = time.time()
            
            # 训练一个batch
            stats = self.train_step(batch_data)
            
            # 记录统计信息
            epoch_stats['actor_losses'].append(stats['actor_loss'])
            epoch_stats['critic_losses'].append(stats['critic_loss'])
            epoch_stats['rewards'].append(stats['avg_reward'])
            epoch_stats['advantages'].append(stats['avg_advantage'])
            epoch_stats['episode_lengths'].append(stats['avg_episode_length'])
            
            batch_time = time.time() - batch_start_time
            
            # 定期打印详细信息
            if batch_idx % self.args.log_interval == 0:
                elapsed_time = time.time() - start_time
                progress = (batch_idx + 1) / total_batches * 100
                
                # 计算ETA
                avg_batch_time = elapsed_time / (batch_idx + 1)
                eta_seconds = avg_batch_time * (total_batches - batch_idx - 1)
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
                # 计算当前学习率
                actor_lr = self.actor_optimizer.param_groups[0]['lr']
                critic_lr = self.critic_optimizer.param_groups[0]['lr']
                
                epoch_prefix = f"Epoch {epoch_num}" if epoch_num is not None else "Training"
                
                logger.info(f"{epoch_prefix} - Batch [{batch_idx+1}/{total_batches}] ({progress:.1f}%)")
                logger.info(f"  ├─ Actor Loss: {stats['actor_loss']:.6f} | Critic Loss: {stats['critic_loss']:.6f}")
                logger.info(f"  ├─ Avg Reward: {stats['avg_reward']:.6f} | Avg Advantage: {stats['avg_advantage']:.6f}")
                logger.info(f"  ├─ Avg Episode Length: {stats['avg_episode_length']:.2f}")
                logger.info(f"  ├─ Learning Rate - Actor: {actor_lr:.2e} | Critic: {critic_lr:.2e}")
                logger.info(f"  ├─ Batch Time: {batch_time:.2f}s | Avg Time: {avg_batch_time:.2f}s/batch")
                logger.info(f"  └─ ETA: {eta_str} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                logger.info("")
            
            # 每个batch后的简单进度提示
            elif batch_idx % (self.args.log_interval // 4) == 0:  # 更频繁的简单进度
                progress = (batch_idx + 1) / total_batches * 100
                logger.info(f"Progress: {progress:.1f}% | Batch {batch_idx+1}/{total_batches} | " f"Reward: {stats['avg_reward']:.4f} | Actor Loss: {stats['actor_loss']:.4f}")
        
        total_time = time.time() - start_time
        
        # 计算epoch平均统计
        avg_stats = {
            'avg_actor_loss': np.mean(epoch_stats['actor_losses']),
            'avg_critic_loss': np.mean(epoch_stats['critic_losses']),
            'avg_reward': np.mean(epoch_stats['rewards']),
            'avg_advantage': np.mean(epoch_stats['advantages']),
            'avg_episode_length': np.mean(epoch_stats['episode_lengths']),
            'std_reward': np.std(epoch_stats['rewards']),
            'max_reward': np.max(epoch_stats['rewards']),
            'min_reward': np.min(epoch_stats['rewards']),
            'total_batches': total_batches,
            'epoch_time': total_time
        }
        
        # 打印epoch总结
        epoch_prefix = f"Epoch {epoch_num}" if epoch_num is not None else "Training Epoch"
        logger.info("=" * 80)
        logger.info(f"{epoch_prefix} Summary:")
        logger.info(f"  ├─ Total Batches: {total_batches} | Total Time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        logger.info(f"  ├─ Avg Actor Loss: {avg_stats['avg_actor_loss']:.6f} | Avg Critic Loss: {avg_stats['avg_critic_loss']:.6f}")
        logger.info(f"  ├─ Avg Reward: {avg_stats['avg_reward']:.6f} ± {avg_stats['std_reward']:.6f}")
        logger.info(f"  ├─ Reward Range: [{avg_stats['min_reward']:.6f}, {avg_stats['max_reward']:.6f}]")
        logger.info(f"  ├─ Avg Advantage: {avg_stats['avg_advantage']:.6f}")
        logger.info(f"  └─ Avg Episode Length: {avg_stats['avg_episode_length']:.2f}")
        logger.info("=" * 80)
        logger.info("")
        
        return avg_stats
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.actor.eval()
        self.critic.eval()
        rewards = []
        all_size = 0
        with torch.no_grad():
            # 处理单个样本的embedding
            for batch_idx, batch_data in enumerate(dataloader):
                state = self.pool.map(self.rfDataTrans.tokenize, [(example, self.rfDataTrans.tokenizer, self.predefined_args) for example in batch_data])
                batch_size = len(batch_data)
                all_size += batch_size
                # 收集轨迹
                all_trajectories = []
                for i in range(batch_size):
                    trajectory = self.collect_episode(
                        crItem=batch_data[i],
                        state=state[i]
                    )
                    all_trajectories.extend(trajectory)
                for step in all_trajectories:
                    if step['is_terminal']:
                        rewards.append(step['reward']) 
                
        if rewards:  # 检查是否有奖励数据
            mean_bleu = np.sum(rewards) / batch_size  # 修正：除以batch_size而不是len(batch_size)
        else:
            mean_bleu = 0.0  # 处理空奖励列表的情况
        self.actor.train()
        self.critic.train()
        return mean_bleu
    
    # 规划train与test的部分
    def run(self, train_dataloader, val_dataloader=None, num_epochs=100):
        """运行训练"""
        logger.info("Starting Simplified RL Training")
        
        best_reward = -float('inf')
        
        for epoch in range(num_epochs):
            # 训练
            train_stats = self.train_epoch(train_dataloader)
            
            # 一个epoch就是一个episode
            self.episode_rewards.append(train_stats['avg_reward'])
            self.actor_losses.append(train_stats['avg_actor_loss'])
            self.critic_losses.append(train_stats['avg_critic_loss'])
            
            # 验证
            if val_dataloader is not None:
                val_reward = self.evaluate(val_dataloader)
                logger.info(f"Epoch {epoch}: "
                           f"Train Reward: {train_stats['avg_reward']:.4f}, "
                           f"Val Reward: {val_reward:.4f}, "
                           f"Actor Loss: {train_stats['avg_actor_loss']:.4f}, "
                           f"Critic Loss: {train_stats['avg_critic_loss']:.4f}")
                
                # 保存最佳模型
                if val_reward > best_reward:
                    best_reward = val_reward
                    self.save_models(epoch, is_best=True)
            else:
                logger.info(f"Epoch {epoch}: "
                           f"Train Reward: {train_stats['avg_reward']:.4f}, "
                           f"Actor Loss: {train_stats['avg_actor_loss']:.4f}, "
                           f"Critic Loss: {train_stats['avg_critic_loss']:.4f}")
            
            # 定期保存
            if epoch % self.args.save_interval == 0:
                self.save_models(epoch)
    
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