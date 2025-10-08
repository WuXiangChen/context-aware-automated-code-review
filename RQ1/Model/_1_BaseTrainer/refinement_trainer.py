import os
import json
import torch
from torch.utils.data import DistributedSampler
from .base_trainer import BaseTrainer
from .utils import RefineDataset, SimpleRefineDataset
from .smooth_bleu import bleu_fromstr, rouge_pp_metrics
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from Model._1_BaseTrainer.configs import set_seed
import logging
import pdb
from transformers import GenerationConfig
from pprint import pprint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)

class RefinementTrainer(BaseTrainer):
    
    def  __init__(self, args, data_file, model, eval_=False, ref_node_list=None):
        super().__init__(args=args, model=model)
        self.args = args
        self.data_file = data_file
        self.eval_ = eval_
        # ref_node_list中存储了全部训练以及测试数据集，给定eval_的情况下，只需要加载其中一部分即可！
        self.len_ = len(ref_node_list)
        if not eval_:
            self.train_ref_node_list = ref_node_list[:int(self.len_*0.8)]
            self.valid_ref_node_list = ref_node_list[int(self.len_*0.8):]
        else:
            self.valid_ref_node_list = ref_node_list
        del ref_node_list
        
    def get_data_loader(self, train_eval_=False):
        # 初始化数据集
        print(self.data_file)
        def fn(features):
            return features
        data_file = []
        if not train_eval_:
            valid_ref_node_list = self.train_ref_node_list
            del self.train_ref_node_list
        else:
            valid_ref_node_list = self.valid_ref_node_list
            del self.valid_ref_node_list
        dataset_ref = SimpleRefineDataset if self.args.raw_input else RefineDataset
        dataset = dataset_ref(self.tokenizer, self.pool,self.args, data_file, ref_node_list=valid_ref_node_list)
        # 创建采样器
        sampler = DistributedSampler(dataset) if not self.eval_ else SequentialSampler(dataset)
        
        return DataLoader(dataset,sampler=sampler,batch_size=self.args.eval_batch_size if self.eval_ else self.args.train_batch_size,num_workers=self.args.cpu_count, collate_fn=fn)
    
    def train_step(self, examples, max_source_length=512, max_target_length=512):
        """
        examples: 输入样本列表
        max_source_length: 输入序列的最大长度
        max_target_length: 输出序列的最大长度
        """
        # 截断或填充输入数据
        def truncate_or_pad(sequence, max_len, pad_id):
            if len(sequence) > max_len:
                return sequence[:max_len]  # 截断超长部分
            else:
                return sequence + [pad_id] * (max_len - len(sequence))  # 填充到固定长度

        # 处理每个样本的 source_ids 和 target_ids
        processed_source_ids = []
        processed_target_ids = []
        for ex in examples:
            # 截断/填充 source_ids
            truncated_source = truncate_or_pad(ex.source_ids, max_source_length, self.tokenizer.pad_id)
            # 截断/填充 target_ids
            truncated_target = truncate_or_pad(ex.target_ids, max_target_length, self.tokenizer.pad_id)
            processed_source_ids.append(truncated_source)
            processed_target_ids.append(truncated_target)

        # 转换为 Tensor 并移动到设备
        source_ids = torch.tensor(processed_source_ids, dtype=torch.long).to(self.local_rank)
        target_ids = torch.tensor(processed_target_ids, dtype=torch.long).to(self.local_rank)

        # 生成掩码（自动处理填充部分）
        source_mask = source_ids.ne(self.tokenizer.pad_id)
        target_mask = target_ids.ne(self.tokenizer.pad_id)

        # 计算损失
        loss = self.model(
            input_ids=source_ids,
            input_labels=None,  # 根据实际需求调整
            decoder_input_ids=target_ids,
            attention_mask=source_mask,
            decoder_attention_mask=target_mask,
            encoder_loss=False
        )
        return loss

    def evaluate(self, dataloader):
        self.model.eval()
        # 首先，这可能和PyTorch的模型并行或数据并行有关。比如，当使用DataParallel或者DistributedDataParallel时，模型会被包裹一层，原来的模型会被放在module属性里。
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
        pred_nls, golds = [], []
        
        with torch.no_grad():
            for examples in dataloader:
                # 生成预测
                source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(self.local_rank)
                preds = model.generate(
                    input_ids=source_ids,
                    attention_mask=source_ids.ne(self.tokenizer.pad_id),
                    use_cache=True,
                    synced_gpus = False,
                    num_beams=self.args.beam_size,
                    early_stopping=True,
                    length_penalty = self.args.length_penalty,
                    max_length=self.args.max_target_length)
                
                # 解码预测结果
                batch_preds = [self.tokenizer.decode(ids[1:],skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) for ids in preds.cpu().numpy()]
                
                batch_golds = [self.tokenizer.decode(ex.target_ids,skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) for ex in examples]
                
                pred_nls.extend(batch_preds)
                golds.extend(batch_golds)

        # 计算BLEU
        return bleu_fromstr(pred_nls, golds, rmstop=False), rouge_pp_metrics(pred_nls, golds)
    
    def run(self):
        try:
            # Initialize training
            global_step = 0
            tr_loss, logging_loss = 0.0, 0.0
            best_bleu = 0.0
            train_dataloader = self.get_data_loader()
            eval_dataloader  = self.get_data_loader(train_eval_=True)
            # Training loop
            for epoch in range(1, self.args.train_epochs + 1):
                # Set seed for reproducible data split
                save_seed = self.args.seed
                self.args.seed += epoch
                set_seed(self.args)
                self.args.seed = save_seed
                self.model.train()
                for step, examples in enumerate(train_dataloader, 1):
                    if step == 1:
                        # ex = examples[0]
                        logger.info(f"batch size: {len(examples)}")
                        # logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                    step+=1
                    # print("examples:",type(examples))
                    loss = self.train_step(examples)
                    if self.args.gpu_per_node > 1:
                        loss = loss.mean()
                    
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    loss.backward()
                    tr_loss += loss.item()
                    
                    # Gradient accumulation and parameter update
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                        
                        # Logging
                        if self.args.global_rank == 0 and global_step % self.args.log_steps == 0:
                            current_loss = (tr_loss - logging_loss) / self.args.log_steps
                            logger.info(
                                f"Epoch: {epoch}, Step: {global_step}/{self.args.train_steps}, "
                                f"Loss: {current_loss:.4f}")
                            logging_loss = tr_loss
                        
                        # Evaluation and model saving
                        if self.args.global_rank == 0 and (
                            global_step % self.args.save_steps == 0 or 
                            global_step == self.args.train_steps):
                            bleu, other_metrics = self.evaluate(eval_dataloader)
                            logger.info(f"Validation Accuracy at step {global_step}: {bleu:.4f}")
                            
                            # Save checkpoint
                            if bleu > best_bleu:
                                best_bleu = bleu
                                output_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}-{bleu:.4f}")
                                os.makedirs(output_dir, exist_ok=True)
                                self.save_model(output_dir, bleu)
                                logger.info(f"New best model saved to {output_dir}")
                    
                    # Early stopping if reach max steps
                    if global_step >= self.args.train_steps:
                        if self.args.global_rank == 0:
                            bleu, other_metrics = self.evaluate(eval_dataloader)
                            output_dir = os.path.join(
                                self.args.output_dir, 
                                f"checkpoint-last-{bleu:.4f}")
                            self.save_model(output_dir, bleu)
                            logger.info(f"Training completed. Final model saved to {output_dir}")
                        return
        finally:
            self.pool.close()
            if self.args.global_rank == 0:
                logger.info("Training finished. Cleaning up resources.")