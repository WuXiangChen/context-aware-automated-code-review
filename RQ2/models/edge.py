# =============================================================================
# models/edge.py
# =============================================================================
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, Union
from utils.logger import setup_logger
logger = setup_logger("logs/"+__name__)

@dataclass
class EdgeInfo:
    """边信息数据类 - 适配MongoDB数据结构"""
    project_name: str
    source_node_id: str
    source_node_type: str  # "Issue" or "PR"
    # source_node_created_at: datetime
    target_node_id: str
    target_node_type: str  # "Issue" or "PR"
    # target_node_created_at: datetime
    edge_created_at: datetime
    mention_type: str  # "issue_pr_reference", etc.
    mention_context: str
    created_at: datetime  # 记录创建时间
    
    @classmethod
    def from_mongo_doc(cls, doc: Dict[str, Any]) -> 'EdgeInfo':
        """从MongoDB文档创建EdgeInfo实例"""
        return cls(
            project_name=doc.get("project_name", ""),
            source_node_id=doc.get("source_node_id", ""),
            source_node_type=doc.get("source_node_type", ""),
            # source_node_created_at=doc.get("source_node_created_at"),
            target_node_id=doc.get("target_node_id", ""),
            target_node_type=doc.get("target_node_type", ""),
            # target_node_created_at=doc.get("target_node_created_at"),
            edge_created_at=doc.get("edge_created_at"),
            mention_type=doc.get("mention_type", ""),
            mention_context=doc.get("mention_context", ""),
            created_at=doc.get("created_at")
        )
    
    @property
    def source_id(self) -> str:
        """获取源节点ID（兼容原接口）"""
        return self.source_node_id
    
    @property
    def target_id(self) -> str:
        """获取目标节点ID（兼容原接口）"""
        return self.target_node_id
    
    @property
    def edge_type(self) -> str:
        """获取边类型（兼容原接口）"""
        return self.mention_type
    
    @property
    def confidence(self) -> float:
        """获取置信度（可基于mention_type计算）"""
        # 根据mention_type计算置信度
        confidence_map = {
            "issue_pr_reference": 0.8,
            "pr_issue_reference": 0.8,
            "closes": 0.9,
            "fixes": 0.9,
            "resolves": 0.9,
            "mentions": 0.6,
            "relates_to": 0.7,
        }
        return confidence_map.get(self.mention_type, 0.5)