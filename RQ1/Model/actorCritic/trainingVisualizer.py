import time
import numpy as np
import os
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import  logging
logger = logging.getLogger("Output/logs/"+__name__)
class TrainingVisualizer:
    """
    独立的训练可视化类，用于记录和可视化训练过程中的各种指标
    """
    
    def __init__(self, 
                 log_dir: str = None,
                 experiment_name: str = None,
                 auto_create_subdir: bool = True,
                 flush_secs: int = 30):
        """
        初始化可视化器
        
        Args:
            log_dir: 日志根目录，默认为 './logs'
            experiment_name: 实验名称，默认为时间戳
            auto_create_subdir: 是否自动创建时间戳子目录
            flush_secs: TensorBoard写入间隔
        """
        self.base_log_dir = log_dir or './Output/rl_logs'
        
        if experiment_name is None:
            experiment_name = f'training_{time.strftime("%Y%m%d_%H%M%S")}'
        
        if auto_create_subdir:
            self.log_dir = os.path.join(self.base_log_dir, experiment_name)
        else:
            self.log_dir = self.base_log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)
        
        # 计数器
        self.global_step = 0
        self.epoch_count = 0
        
        # 数据缓存，用于计算统计信息
        self.epoch_cache = defaultdict(list)
        self.batch_cache = defaultdict(list)
        
        # 配置项
        self.config = {
            'log_gradients': False,
            'log_weights': False,
            'log_distributions': True,
            'log_custom_plots': True,
            'moving_average_window': 10,
        }
        
        print(f"✓ TensorBoard logs initialized at: {self.log_dir}")
        print(f"✓ Run 'tensorboard --logdir {self.log_dir}' to view logs")
    
    def configure(self, **kwargs):
        """配置可视化选项"""
        self.config.update(kwargs)
        return self
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录标量值"""
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag_dict: Dict[str, float], step: Optional[int] = None):
        """批量记录标量值"""
        if step is None:
            step = self.global_step
        for tag, value in tag_dict.items():
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: Union[np.ndarray, torch.Tensor, List], step: Optional[int] = None):
        """记录直方图"""
        if step is None:
            step = self.global_step
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, list):
            values = np.array(values)
        self.writer.add_histogram(tag, values, step)
    
    def log_batch_metrics(self, metrics: Dict[str, Any], batch_idx: int = None, prefix: str = "Batch"):
        """
        记录批次级别的指标
        
        Args:
            metrics: 指标字典，例如 {'actor_loss': 0.1, 'reward': 1.5}
            batch_idx: 批次索引
            prefix: 标签前缀
        """
        step = self.global_step if batch_idx is None else self.global_step
        
        # 记录标量指标
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                tag = f"{prefix}/{key.replace('_', ' ').title()}"
                self.writer.add_scalar(tag, value, step)
                
                # 缓存数据用于epoch统计
                self.epoch_cache[key].append(value)
        
        self.global_step += 1
    
    def log_epoch_summary(self, epoch: int, custom_metrics: Dict[str, Any] = None):
        """
        记录epoch总结统计
        
        Args:
            epoch: epoch编号
            custom_metrics: 额外的自定义指标
        """
        self.epoch_count = epoch
        
        # 计算epoch统计
        epoch_stats = {}
        for key, values in self.epoch_cache.items():
            if values:  # 确保列表不为空
                values_array = np.array(values)
                epoch_stats.update({
                    f"Epoch/Average_{key.title()}": np.mean(values_array),
                    f"Epoch/Std_{key.title()}": np.std(values_array),
                    f"Epoch/Max_{key.title()}": np.max(values_array),
                    f"Epoch/Min_{key.title()}": np.min(values_array),
                })
                
                # 记录分布直方图
                # if self.config['log_distributions']:
                #     self.writer.add_histogram(f"Distribution/{key.title()}", values_array, epoch)
        
        # 记录epoch统计到TensorBoard
        for tag, value in epoch_stats.items():
            self.writer.add_scalar(tag, value, epoch)
        
        # 记录自定义指标
        if custom_metrics:
            for key, value in custom_metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.writer.add_scalar(f"Custom/{key}", value, epoch)
        
        # 创建自定义图表
        if self.config['log_custom_plots']:
            self._log_custom_plots(epoch)
        
        # 清空缓存
        self.epoch_cache.clear()
    
    def log_learning_rates(self, optimizers: Dict[str, torch.optim.Optimizer], step: Optional[int] = None):
        """记录学习率"""
        if step is None:
            step = self.global_step
            
        for name, optimizer in optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                tag = f"Learning_Rate/{name}" + (f"_Group_{i}" if len(optimizer.param_groups) > 1 else "")
                self.writer.add_scalar(tag, lr, step)
    
    def log_model_weights(self, model: torch.nn.Module, model_name: str, step: Optional[int] = None):
        """记录模型权重分布"""
        if not self.config['log_weights']:
            return
            
        if step is None:
            step = self.epoch_count
            
        for name, param in model.named_parameters():
            if param.data is not None:
                self.writer.add_histogram(f"{model_name}_Weights/{name}", 
                                        param.data.cpu().numpy(), step)
    
    def log_model_gradients(self, model: torch.nn.Module, model_name: str, step: Optional[int] = None):
        """记录模型梯度"""
        if not self.config['log_gradients']:
            return
            
        if step is None:
            step = self.global_step
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data.cpu().numpy()
                self.writer.add_histogram(f"{model_name}_Gradients/{name}", grad_data, step)
                self.writer.add_scalar(f"{model_name}_Gradient_Norm/{name}", 
                                     np.linalg.norm(grad_data), step)
    
    def log_training_progress(self, 
                            current_batch: int, 
                            total_batches: int, 
                            epoch: int,
                            batch_time: float = None,
                            eta_seconds: float = None):
        """记录训练进度"""
        progress = (current_batch + 1) / total_batches * 100
        step = self.global_step
        
        self.writer.add_scalar('Training/Progress_Percent', progress, step)
        self.writer.add_scalar('Training/Current_Epoch', epoch, step)
        self.writer.add_scalar('Training/Current_Batch', current_batch, step)
        
        if batch_time is not None:
            self.writer.add_scalar('Training/Batch_Time', batch_time, step)
        
        if eta_seconds is not None:
            self.writer.add_scalar('Training/ETA_Minutes', eta_seconds / 60, step)
    
    def log_hyperparameters(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float] = None):
        """记录超参数"""
        # 过滤出可序列化的超参数
        filtered_hparams = {}
        for key, value in hparam_dict.items():
            if isinstance(value, (int, float, str, bool)):
                filtered_hparams[key] = value
            elif hasattr(value, '__name__'):  # 函数或类
                filtered_hparams[key] = str(value.__name__)
            else:
                filtered_hparams[key] = str(value)
        
        self.writer.add_hparams(filtered_hparams, metric_dict or {})
    
    def _log_custom_plots(self, epoch: int):
        """创建和记录自定义图表"""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')  # 避免样式冲突
            
            # 如果有reward数据，创建趋势图
            if 'avg_reward' in self.epoch_cache or 'reward' in self.epoch_cache:
                reward_key = 'avg_reward' if 'avg_reward' in self.epoch_cache else 'reward'
                rewards = self.epoch_cache.get(reward_key, [])
                
                if len(rewards) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 原始数据
                    ax.plot(rewards, 'b-', alpha=0.6, linewidth=1, label='Batch Reward')
                    
                    # 移动平均
                    window = min(self.config['moving_average_window'], len(rewards))
                    if window > 1:
                        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        x_ma = range(window-1, len(rewards))
                        ax.plot(x_ma, moving_avg, 'r-', linewidth=2, label=f'Moving Average (w={window})')
                    
                    ax.set_xlabel('Batch')
                    ax.set_ylabel('Reward')
                    ax.set_title(f'Reward Trend - Epoch {epoch}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    self.writer.add_figure('Custom_Plots/Reward_Trend', fig, epoch)
                    plt.close(fig)
            
            # 如果有损失数据，创建损失对比图
            loss_keys = [key for key in self.epoch_cache.keys() if 'loss' in key.lower()]
            if len(loss_keys) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for key in loss_keys[:4]:  # 最多显示4个损失
                    values = self.epoch_cache[key]
                    ax.plot(values, label=key.replace('_', ' ').title(), alpha=0.8)
                
                ax.set_xlabel('Batch')
                ax.set_ylabel('Loss')
                ax.set_title(f'Loss Comparison - Epoch {epoch}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')  # 使用对数坐标
                
                self.writer.add_figure('Custom_Plots/Loss_Comparison', fig, epoch)
                plt.close(fig)
                
        except ImportError:
            pass  # matplotlib不可用时跳过
        except Exception as e:
            print(f"Warning: Could not create custom plots: {e}")
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """记录文本信息"""
        if step is None:
            step = self.global_step
        self.writer.add_text(tag, text, step)
    
    def log_image(self, tag: str, img_tensor: torch.Tensor, step: Optional[int] = None):
        """记录图像"""
        if step is None:
            step = self.global_step
        self.writer.add_image(tag, img_tensor, step)
    
    def flush(self):
        """手动刷新到磁盘"""
        self.writer.flush()
    
    def close(self):
        """关闭writer并清理资源"""
        self.writer.close()
        print(f"✓ TensorBoard logs saved to: {self.log_dir}")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        self.close()