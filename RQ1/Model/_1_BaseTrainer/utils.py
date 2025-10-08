import re, json
import os, random
import torch, logging
from copy import deepcopy as cp
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import T5Tokenizer, RobertaTokenizer
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    RobertaTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer)
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MyTokenizer(object):
    """
    Wrapper for ByteLevelBPETokenizer
    """
    def __init__(self, vocab=None, merges=None, **kwargs):
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges, **kwargs)
        self.update_id2token()

    @staticmethod
    def from_pretrained(path):
        vocabp = os.path.join(path, "vocab.json")
        mergesp = os.path.join(path, "merges.txt")
        mytoken = MyTokenizer(vocabp, mergesp)
        return mytoken

    def update_id2token(self):
        vocab = self.tokenizer.get_vocab()
        self.id2token = {vocab[token]: token for token in vocab}

    def add_special_tokens(self, dic):
        for values in dic.values():
            self.tokenizer.add_special_tokens(values)
        self.update_id2token()

    def convert_ids_to_tokens(self, ids):
        vocab = self.id2token
        return [vocab[i] for i in ids]
    
    def decode(self, ids, **kwargs):    ##### to be update
        tokens = self.convert_ids_to_tokens(ids)
        return " ".join(tokens)

    def encode(self, text, **kwargs):
        text = text.encode("ascii", errors="ignore").decode("ascii")
        return self.tokenizer.encode(text).ids

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.tokenizer.get_vocab())

# class RefineFeatures(object):
#     def __init__(self, example_id, source_ids, target_ids, srcMsg, trgMsg):
#         self.example_id = example_id
#         self.source_ids = source_ids
#         self.target_ids = target_ids
#         self.srcMsg = srcMsg
#         self.trgMsg = trgMsg
        
#     def get_combined_embedding(self):
#         """拼接 example_id + source_ids + target_ids 为一个向量"""
#         # 将 example_id 转为单元素列表，并和其他向量拼接
#         return [float(self.example_id)] + self.source_ids + self.target_ids

class RefineFeatures(object):
    def __init__(self, example_id, source_ids, target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        
    def get_combined_embedding(self):
        """拼接 example_id + source_ids + target_ids 为一个向量"""
        # 将 example_id 转为单元素列表，并和其他向量拼接
        return [float(self.example_id)] + self.source_ids + self.target_ids

class RefineDataset(Dataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1, ref_node_list=None):
        self.tokenizer = tokenizer
        self.args = args
        if ref_node_list is not None:
            print("Reading examples from ref_node_list, its length {} and type is {}".format(len(ref_node_list), type(ref_node_list)))
        # 在原有的架构中，这里是加载数据的必要部分
        ## 现在需要将其更改成nodeclass的格式，保持最小的改动，这里的引用格式最好能与现有格式保持一致
        # examples = [json.loads(line) for line in open(file_path)]
        examples = ref_node_list
        if examples is not None:
            for i in range(len(examples)):
                examples[i]["id"] = i
            if samplenum > 0:
                examples = examples[:samplenum]
            logger.info(f"Tokenize examples: {file_path}")
            if self.args.task_type=="msg":
                self.feats = pool.map(self.tokenize, [(example, tokenizer, args) for example in examples])
            elif self.args.task_type=="ref":
                self.feats = pool.map(self.tokenize_for_ref, [(example, tokenizer, args) for example in examples])
        
    def tokenize(self, item):
        example, tokenizer, args = item
        
        oldf = example["oldf"].split("\n")
        codediff = example["codediff"].split("\n")
        oldlines = codediff
        oldlines = [line[1:].strip() for line in oldlines]
        oldlines = "\n".join(oldlines)
        oldlines = "<add>" + oldlines.replace("\n", "<add>")
        
        # 这里的输入改成 context
        title_and_desc = example["context"]
        title_and_desc = "<add>" + title_and_desc.replace("\n", "<add>")
        srcids = self.encode_remove(tokenizer, oldlines, args)
        srcids += [tokenizer.msg_id] + self.encode_remove(tokenizer, title_and_desc, args)
        
        # 这里的输出改成 comment
        comment = example["comment"]
        tgtids = [tokenizer.msg_id] + self.encode_remove(tokenizer, comment, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return RefineFeatures(example["id"], srcids, tgtids)
    
    # def tokenize(self, item, firstFlag=False):
    #     example, tokenizer, args = item
    #     example["id"] = example["repo_id"]
    #     oldlines = example["codediff"].split("\n")
    #     oldlines = [line[1:].strip() for line in oldlines]
    #     oldlines = "\n".join(oldlines)
    #     oldlines = "<add>" + oldlines.replace("\n", "<add>")
    #     oldsLine_srcids = self.encode_remove(tokenizer, oldlines, args.max_source_length//2)
    #     oldLine_ids = self.pad_sequence(oldsLine_srcids, max_length=args.max_source_length//2, tokenizer=tokenizer)
    #     context = example["context"]
    #     context = "<add>" + context.replace("\n", "<add>")
    #     context_srcids = self.encode_remove(tokenizer, context, args.max_source_length//2-1)
    #     context_ids = self.pad_sequence(context_srcids, max_length=args.max_source_length//2-1, tokenizer=tokenizer)
    #     srcids = oldLine_ids + [tokenizer.msg_id] + context_ids

    #     comment = example["comment"]
    #     tgtids = [tokenizer.msg_id] + self.encode_remove(tokenizer, comment, args.max_source_length)   
    #     tgtids = self.pad_sequence(tgtids, max_length=args.max_source_length, tokenizer=tokenizer)
    #     srcMsg = context + oldlines
    #     trgMsg = comment
    #     return RefineFeatures(example["id"], srcids, tgtids, srcMsg, trgMsg)
    
    
    def tokenize_for_ref(self, item):
        example, tokenizer, args = item
        oldlines = example["old"].split("\n")
        newlines = example["new"].split("\n")
        oldlines = [line[1:].strip() for line in oldlines]
        newlines = [line[1:].strip() for line in newlines]
        oldlines = "\n".join(oldlines)
        newlines = "\n".join(newlines)
        oldlines = "<add>" + oldlines.replace("\n", "<add>")
        newlines = "<add>" + newlines.replace("\n", "<add>")
        comment = example["comment"]
        srcids = self.encode_remove(tokenizer, oldlines, args)
        # print("tokenizer.msg_id:",str(tokenizer.msg_id))
        srcids += [tokenizer.msg_id] + self.encode_remove(tokenizer, comment, args)
        tgtids = self.encode_remove(tokenizer, newlines, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return RefineFeatures(example["id"], srcids, tgtids)
    
    @staticmethod
    def process_pred_gold(pred, gold):
        gold = gold.split("\n")
        gold = [line[1:].strip() for line in gold]
        gold = " ".join(gold)
        pred = " ".join(pred.split())
        pred = pred.replace("<add> ", "")
        return pred, gold
    
    
    def pad_sequence(self, ids, max_length, tokenizer):
        """标准填充到固定长度"""
        ids = ids[:max_length - 2]  # 预留BOS和EOS位置
        ids = [tokenizer.bos_id] + ids + [tokenizer.eos_id]
        pad_len = max_length - len(ids)
        padded_ids = ids + [tokenizer.pad_id] * pad_len
        assert len(padded_ids) == max_length, "Not equal length."
        return padded_ids
    
    
    def pad_assert(self, source_ids, target_ids, args, tokenizer):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return source_ids, target_ids
    
    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text, max_length=args.max_source_length, truncation=True)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        elif type(tokenizer) == MyTokenizer:
            return text
        else:
            raise NotImplementedError
    # def pad_assert(self, source_ids, target_ids, args, tokenizer):
    #     source_ids = source_ids[:args.max_source_length - 2]
    #     source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
    #     pad_len = args.max_source_length - len(source_ids)
    #     source_ids += [tokenizer.pad_id] * pad_len
    #     target_ids = target_ids[:args.max_target_length - 2]
    #     target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
    #     pad_len = args.max_target_length - len(target_ids)
    #     target_ids += [tokenizer.pad_id] * pad_len
    #     assert len(source_ids) == args.max_source_length, "Not equal length."
    #     assert len(target_ids) == args.max_target_length, "Not equal length."
    #     return source_ids, target_ids

    # def encode_remove(self, tokenizer, text, max_source_length):
    #     text = tokenizer.encode(text, max_length=max_source_length, truncation=True)
    #     if type(tokenizer) == T5Tokenizer:
    #         return text[:-1]
    #     elif type(tokenizer) == RobertaTokenizer:
    #         return text[1:-1]
    #     elif type(tokenizer) == MyTokenizer:
    #         return text
    #     else:
    #         raise NotImplementedError
    

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

class SimpleRefineDataset(RefineDataset):
    def tokenize(self, item):
        example, tokenizer, args = item
        oldlines = example["old"].split("\n")
        newlines = example["new"].split("\n")
        oldlines = [line[1:].strip() for line in oldlines]
        newlines = [line[1:].strip() for line in newlines]
        oldlines = " ".join(oldlines)
        newlines = " ".join(newlines)
        comment = example["comment"]
        srcids = self.encode_remove(tokenizer, oldlines, args)
        srcids += [tokenizer.msg_id] + self.encode_remove(tokenizer, comment, args)
        tgtids = self.encode_remove(tokenizer, newlines, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return RefineFeatures(example["id"], srcids, tgtids)

    @staticmethod
    def process_pred_gold(pred, gold):
        gold = gold.split("\n")
        gold = [line[1:].strip() for line in gold]
        gold = " ".join(gold)
        pred = " ".join(pred.split())
        return pred, gold

# 这部分的信息都是需要根据不同的模型显式的注入
def build_or_load_gen_model(args, model):
    config_class, model_class, tokenizer_class = model.config_class, model.model_class, model.tokenizer_class
    logger.info("==========")
    if os.path.exists(args.model_name_or_path) and args.load_model_path is None:
        logger.info(f"loaded model path:{args.model_name_or_path}")
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        # 这里打印model_class的在embedding层的参数大小
        logger.info(f"开始定义模型")
        model = model_class.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=args.model_name_or_path)
        # from transformers import T5Model
        # model = T5Model.from_pretrained("google-t5/t5-small")
    elif args.load_model_path is not None:
        logger.info(f"loaded model path:{args.load_model_path}")
        config = config_class.from_pretrained(args.load_model_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        # 这里打印model_class的在embedding层的参数大小
        logger.info(f"开始定义模型")
        model = model_class.from_pretrained(pretrained_model_name_or_path=args.load_model_path, config=args.load_model_path)
    else:
        # 报错
        raise ValueError("No model path provided. Please specify --model_name_or_path or --load_model_path.")
    
    # 第一个问题，这里的tokenizer是如何定义的？从哪里继承的？它的<e{i}>的符号是什么意思。
    tokenizer.special_dict = {f"<e{i}>" : tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)}

    # 加入特殊token，这些特殊的字符应该都是预定义的。
    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    # 这里判断model的embedding层大小是是否合适，如果不合适则手动扩展
    if hasattr(model, "get_input_embeddings"):
        old_num_tokens = model.get_input_embeddings().weight.size(0)
        new_num_tokens = len(tokenizer)
        if old_num_tokens != new_num_tokens:
            logger.info(f"Resizing token embeddings from {old_num_tokens} to {new_num_tokens}")
            # 这里是直接拓展，丢失了部分预训练的内容？
            model.resize_token_embeddings(new_num_tokens)
        if hasattr(model, "config"):
            model.config.vocab_size = new_num_tokens
        if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
            model.encoder.config.vocab_size = new_num_tokens

    return config, model, tokenizer

