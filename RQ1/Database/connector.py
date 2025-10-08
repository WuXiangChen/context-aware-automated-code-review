# =============================================================================
# database/connector.py
# =============================================================================
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from typing import List, Optional, Set
from Database.settings import DatabaseConfig
from Utils.logger import setup_logger
from .query_set import get_all_valid_ref_node_query, get_all_ref_node_query
from .node import CodeReviewItem
from tqdm import tqdm
import copy
logger = setup_logger("Output/logs/"+__name__)

class MongoDBConnector:
    """MongoDB连接管理器"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
    
    def connect(self) -> bool:
        """建立MongoDB连接"""
        try:
            self.client = MongoClient(self.config.connection_string)
            self.db = self.client[self.config.database_name]
            # 测试连接
            self.client.admin.command('ping')
            logger.info(f"成功连接到MongoDB数据库: {self.config.database_name}")
            return True
        except Exception as e:
            logger.error(f"连接MongoDB失败: {e}")
            return False

    
    def get_only_CodeReviewerDatasetREF(self, graph_dict):
        re = self.db["CodeReviewerDatasetREF"].aggregate(get_all_ref_node_query)
        try:
            total = self.db["CodeReviewerDatasetREF"].count_documents({})
        except:
            total = None  # 如果不能获取总数，进度条将显示迭代次数
        
        key_list = [item["node_id"] for item in graph_dict] # 按序得到node_id的结果
        
        all_context_info = []
        for doc in tqdm(re, total=total, desc="Processing documents"):
            repo_id = doc["repo_id"]
            if repo_id not in key_list:
                continue
            else:
                _index = key_list.index(repo_id)
                graph_item = copy.deepcopy(graph_dict[_index])
                graph_item.update(doc)
                all_context_info.append(graph_item)
        return [CodeReviewItem.from_dict(doc) for doc in tqdm(all_context_info, total=len(all_context_info), desc="Generating Final Context Graph")]
        
        
    def disconnect(self):
        """断开MongoDB连接"""
        if self.client:
            self.client.close()
            logger.info("MongoDB连接已断开")
    
    def get_collection(self, collection_name: str) -> Collection:
        """获取指定集合"""
        if self.db is None:
            raise Exception("数据库未连接")
        return self.db[collection_name]
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        
    def close(self):
        """关闭连接"""
        self.disconnect()