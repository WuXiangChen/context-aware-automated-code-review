import os
from safetensors import safe_open
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    RobertaTokenizer)
import logging

logger = logging.getLogger(__name__)
class contextAwareModel(T5ForConditionalGeneration):
    """
    基于 CodeT5 的 Context-Aware 模型
    - 增加 codediff -> context 的 cross-attention
    - 输出与原始编码器输出 residual 相加
    """

    def __init__(self, config, args=None):
        if config is not None:
            super().__init__(config)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=4,
                batch_first=True
            )
            # 添加升维层
            self.upsample_layer = nn.Linear(config.d_model, config.d_model * 2)
            self.layer_norm = nn.LayerNorm(config.d_model)
            # gating 参数
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.init_layer()
        
        self.args = args
        self._init_model_classes()
    
    def init_layer(self):
        """初始化模型权重"""
        # 初始化升维层
        init.kaiming_uniform_(self.upsample_layer.weight, nonlinearity='relu')
        init.zeros_(self.upsample_layer.bias)
        
        # 初始化 Cross Attention
        init.xavier_uniform_(self.cross_attention.in_proj_weight)
        init.zeros_(self.cross_attention.in_proj_bias)
        init.xavier_uniform_(self.cross_attention.out_proj.weight)
        init.zeros_(self.cross_attention.out_proj.bias)

        # LayerNorm 通常初始化为 1 和 0
        init.ones_(self.layer_norm.weight)
        init.zeros_(self.layer_norm.bias)

        
    
    def _init_model_classes(self):
        """动态设置模型相关类"""
        self.config_class = getattr(self.args, 'config_class', T5Config)
        self.model_class = getattr(self.args, 'model_class', contextAwareModel)
        self.tokenizer_class = getattr(self.args, 'tokenizer_class', RobertaTokenizer)

    def forward(self, *argv, **kwargs):
        
        if "input_labels" in kwargs:
            assert ("input_ids" in kwargs and "input_labels" in kwargs and "decoder_input_ids" in kwargs and "attention_mask" in kwargs and "decoder_attention_mask" in kwargs),\
            "Please give these arg keys."
            input_ids = kwargs["input_ids"]
            input_labels = kwargs["input_labels"]
            decoder_input_ids = kwargs["decoder_input_ids"]
            attention_mask = kwargs["attention_mask"]
            decoder_attention_mask = kwargs["decoder_attention_mask"]
            if "encoder_loss" not in kwargs:
                encoder_loss = True
            else:
                encoder_loss = kwargs["encoder_loss"]
            return self.review_forward(input_ids, input_labels, decoder_input_ids, attention_mask, decoder_attention_mask, encoder_loss)
        return super().forward(*argv, **kwargs)

    def review_forward(
        self,
        input_ids,
        input_labels,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        encoder_loss=True
    ):
        # 1. 同时编码 codediff 和 context
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False)
        hidden_states = encoder_outputs[0] # 取张量的第一个元素，作为hidden_state
        batch_size = input_ids.size(0)
        # 分离 codediff 和 context 的隐藏状态
        codediff_len = input_ids.size(1)
        codediff_hidden = hidden_states[:, :codediff_len//2, :]
        context_hidden = hidden_states[:, codediff_len//2:, :]
        context_attention_mask = attention_mask[:, codediff_len//2:]
        
        # 2. cross-attention
        cross_out, _ = self.cross_attention(
            query=codediff_hidden,
            key=context_hidden,
            value=context_hidden
        )
        # 3. 对cross_out进行升维
        cross_out_upsampled = self.upsample_layer(cross_out)  # [batch_size, seq_len, d_model*2]
        cross_out_upsampled = cross_out_upsampled.reshape(batch_size, codediff_len, -1)
        fused_hidden = cross_out_upsampled + hidden_states
        
        # 5. 解码器 - 使用拼接后的fused_hidden作为encoder_hidden_states
        decoder_inputs = self._shift_right(decoder_input_ids)
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_hidden,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False)
        
        # 6. 输出和损失计算
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings: # this is True default
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        
        if encoder_loss:
            cls_logits = nn.functional.linear(hidden_states, self.encoder.get_input_embeddings().weight)
        
        lm_logits = self.lm_head(sequence_output)
        
        if decoder_input_ids is not None:
            lm_loss_fct = CrossEntropyLoss(ignore_index=0)      # Warning: PAD_ID should be 0
            loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))
            if encoder_loss and input_labels is not None:
                cls_loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss += cls_loss_fct(cls_logits.view(-1, cls_logits.size(-1)), input_labels.view(-1))
            return loss
        return cls_logits, lm_logits