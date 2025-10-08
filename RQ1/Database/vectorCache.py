import hashlib
import chromadb
from chromadb.config import Settings
from typing import List
from Model._1_BaseTrainer.utils import RefineFeatures
class VectorCache:
    def __init__(self, collection_name="vector_cache"):
        # 初始化Chroma客户端（持久化存储）
        self.client = chromadb.PersistentClient(path="./Data/vectorCache")  # 指定存储路径
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
    
    def query_or_generate(self, text_hash, cr_item, generate_fn):
        """
        先查询向量数据库，未命中则调用生成函数
        :param text: 原始文本
        :param generate_fn: 生成函数（返回向量）
        :return: 向量结果
        """
        
        # 1. 先尝试查询缓存
        results = self.collection.get(ids=[text_hash], include=["embeddings"])
        if results['ids']:  # 命中缓存
            # print(f"Cache hit for {text_hash}")
            return self.decode_to_refinefeatures(results['embeddings'][0])
        # if text_hash=="539b6738caa6d2dff406cc45d2a31ac0a2721e2e32aff2c0f5648d5d5409c5a3":
        #     print(0)
        # 2. 未命中则生成向量
        # print(f"Cache miss, generating for {text_hash}")
        refine_feat = generate_fn(cr_item)
        
        # 3. 存储到向量数据库
        self.collection.add(
            ids=[text_hash],
            embeddings=[refine_feat.get_combined_embedding()]
        )
        return refine_feat
    
    def decode_to_refinefeatures(self, combined_vector: List[float]) -> RefineFeatures:
        """将拼接后的向量解码为 RefineFeatures 对象"""
        # 假设已知 source_ids 和 target_ids 的原始长度
        source_len = (len(combined_vector)-1) // 2
        
        # 解析各部分数据
        example_id = int(combined_vector[0])          # 第一个元素是 example_id
        source_ids = combined_vector[1:1+source_len]  # 后续是 source_ids
        target_ids = combined_vector[1+source_len:]   # 剩余是 target_ids
        
        return RefineFeatures(
            example_id=example_id,
            source_ids=source_ids,
            target_ids=target_ids
        )