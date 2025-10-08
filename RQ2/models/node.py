from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from utils.logger import setup_logger
logger = setup_logger("logs/"+__name__)

@dataclass
class NodeInfo:
    """节点信息数据类 - 适配MongoDB数据结构"""
    node_id: str  # 从 "repo/owner:NUM" 格式生成，如 "amphtml/ampproject:1"
    number: int
    title: str
    state: str  # "OPEN", "CLOSED", "MERGED"
    created_at: datetime  # 从 "createdAt" 字段解析
    updated_at: datetime  # 从 "updatedAt" 字段解析
    closed_at: Optional[datetime]  # 从 "closedAt" 字段解析
    body: str
    author: str
    labels: List[str]
    node_type: str  # 'issue' or 'pull_request'
    owner_repo: str  # 从 "repo/owner" 字段获取
    
    @classmethod
    def from_mongo_doc(cls, doc: Dict[str, Any]) -> 'NodeInfo':
        """从MongoDB文档创建NodeInfo实例"""
        return cls(
            node_id=doc.get("owner/repo:NUM", ""),
            number=doc.get("number", 0),
            title=doc.get("title", ""),
            state=doc.get("state", ""),
            created_at=doc.get("createdAt"),
            updated_at=doc.get("updatedAt"),
            closed_at=doc.get("closedAt") if doc.get("closedAt") else None,
            body=doc.get("body", ""),
            author=doc.get("author", ""),
            labels=doc.get("labels", []),
            node_type=doc.get("type", ""),
            owner_repo=doc.get("owner/repo", "")
        )
    
    @property
    def project_id(self) -> str:
        """获取项目ID（兼容原接口）"""
        return self.owner_repo