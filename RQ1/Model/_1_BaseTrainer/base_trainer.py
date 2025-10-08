import os
import torch
import logging
import argparse
import random
import json
import multiprocessing
from datetime import timedelta
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup
from .utils import build_or_load_gen_model
from .configs import set_dist, set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# 这里基本思路虽然是没问题的，但是和codereviewer绑定的太过紧了，不适合一般工作的训练泛化
class BaseTrainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self._init_distributed()
        self._load_components()
        self._setup_training()

    def _init_distributed(self): # 这里主要是对分布式训练的必要参数进行初始化
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.args.rank = int(os.environ['RANK'])
            self.args.world_size = int(os.environ['WORLD_SIZE'])
            self.args.gpu = self.args.local_rank = int(os.environ['LOCAL_RANK'])
    
        elif hasattr(self.args, 'gpu_per_node'):
            self.args.rank = self.args.local_rank  # 假设 args.local_rank 已定义
            self.args.world_size = self.args.gpu_per_node
            
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
        self.local_rank = dist.get_rank() % self.args.gpu_per_node
        self.args.global_rank = self.local_rank + self.args.node_index * self.args.gpu_per_node
        self.args.local_rank = self.local_rank
        self.args.world_size = dist.get_world_size()
        torch.cuda.set_device(self.local_rank)
        logger.warning(f"Process rank: {self.local_rank}, global rank: {self.args.global_rank}, world size: {self.args.world_size}")

    def _load_components(self): # 这里主要是进行基础模型的加载，这一点没问题，可以在不同的模型中共享
        set_seed(self.args) # 这里必须指定所有关于args的参数，也就是说所有的专有模型应该维护一个自己的config和tokenizer
        self.config, self.model, self.tokenizer = build_or_load_gen_model(self.args, self.model) 
        self.model = DDP(self.model.cuda(), device_ids=[self.local_rank], find_unused_parameters=True)
        set_dist(self.args)
        self.pool = multiprocessing.Pool(self.args.cpu_count)

    def _setup_training(self): # 初始化训练参数，与基本训练机制
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(self.args.train_steps*0.1),
            num_training_steps=self.args.train_steps)
        self._load_checkpoint()

    def _load_checkpoint(self):
        """加载检查点，包括模型、优化器和调度器状态"""
        checkpoint_dir = f"{self.args.output_dir}/checkpoints-last/"
        
        if not os.path.exists(checkpoint_dir):
            logger.info(f"Checkpoint directory not found: {checkpoint_dir}")
            return False
        
        try:
            # 方案1：如果使用safetensors格式
            model_path = os.path.join(checkpoint_dir, "model.safetensors")
            if os.path.exists(model_path):
                logger.info(f"Loading model from safetensors: {model_path}")
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict)
                logger.info("Model state loaded successfully from safetensors")
            else:
                # 方案2：尝试加载pytorch格式
                model_path = os.path.join(checkpoint_dir, "model.pt")
                if os.path.exists(model_path):
                    logger.info(f"Loading model from pytorch format: {model_path}")
                    state_dict = torch.load(model_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    logger.info("Model state loaded successfully from pytorch format")
                else:
                    logger.warning("No model file found (neither .safetensors nor .pt)")
                    return False
            
            # 加载优化器状态
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                logger.info(f"Loading optimizer state: {optimizer_path}")
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                self.optimizer.load_state_dict(optimizer_state)
                logger.info("Optimizer state loaded successfully")
            else:
                logger.warning("Optimizer state file not found, using fresh optimizer")
            
            # 加载调度器状态
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            if os.path.exists(scheduler_path):
                logger.info(f"Loading scheduler state: {scheduler_path}")
                scheduler_state = torch.load(scheduler_path, map_location="cpu")
                self.scheduler.load_state_dict(scheduler_state)
                logger.info("Scheduler state loaded successfully")
            else:
                logger.warning("Scheduler state file not found, using fresh scheduler")
            
            # 加载训练状态（可选）
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(training_state_path):
                logger.info(f"Loading training state: {training_state_path}")
                training_state = torch.load(training_state_path, map_location="cpu")
                
                # 恢复训练状态
                self.current_epoch = training_state.get('epoch', 0)
                self.global_step = training_state.get('global_step', 0)
                self.best_metric = training_state.get('best_metric', float('-inf'))
                
                logger.info(f"Resuming from epoch: {self.current_epoch}")
                logger.info(f"Global step: {self.global_step}")
                logger.info(f"Best metric: {self.best_metric}")
            
            logger.info("All checkpoint states loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error("Will continue with fresh initialization")
            return False

    def save_model(self, output_dir, metric_value): # 模型保存
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    # 开放train_step，data_loader和evaluate以及具体的run给子类implementation
    def get_data_loader(self, data_file, eval=False):
        raise NotImplementedError

    def train_step(self, examples):
        raise NotImplementedError

    def evaluate(self, dataloader):
        raise NotImplementedError
    
    def run(self):
        try:
            # Training loop implementation
            pass
        finally:
            self.pool.close()