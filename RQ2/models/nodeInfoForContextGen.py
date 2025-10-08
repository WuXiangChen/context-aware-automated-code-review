from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from models.node import NodeInfo

@dataclass
class Comment:
    """评论信息"""
    author: str
    date: str
    content: str
    
    @classmethod
    def from_dict(cls, data: dict):
        content = data["payload"]["comment"]["body"]
        author = data["payload"]["comment"]["user"]["login"]
        date = data["payload"]["comment"]["created_at"]
        return cls(author=author, date=date, content=content)
    
@dataclass
class NodeCInfo:
    """开源软件制品的过程信息"""
    # Title信息
    artInfo: NodeInfo
    
    # Comments信息
    comments: List[Comment] = field(default_factory=list)
    
    # 其他可选字段
    status: Optional[str] = None
    milestone: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    
    def add_comment(self, author: str, date: datetime, content: str):
        """添加评论"""
        comment = Comment(author=author, date=date, content=content)
        self.comments.append(comment)
    
    def get_comments_by_author(self, author: str) -> List[Comment]:
        """获取指定作者的所有评论"""
        return [comment for comment in self.comments if comment.author == author]
    
    def get_latest_comment(self) -> Optional[Comment]:
        """获取最新评论"""
        if not self.comments:
            return None
        return max(self.comments, key=lambda c: c.date)
    
    def __str__(self) -> str:
        """Format NodeCInfo as a structured string representation"""
        result = []
        
        # Header with issue number
        result.append(f"**Artifact Id {self.artInfo.number}:**")
        
        # Title section
        result.append("l  **Title**")
        title_line = (f"n  {self.artInfo.author}, opened on "
                    f"{self.artInfo.created_at}: "
                    f"{self.artInfo.title} #{self.artInfo.number}.")
        result.append(title_line)
        
        # Current behavior section (description)
        result.append("l  **Current behavior And Expected behavior**")
        # Split description into lines and add proper indentation
        description_lines = self.artInfo.body.split('\n')
        for line in description_lines:
            if line.strip():  # Only add non-empty lines
                result.append(f"n  {line}")
        # Conversation section
        if self.comments:
            result.append("l  **Conversation**")
            for comment in sorted(self.comments, key=lambda c: c.date):
                comment_line = (f"n  {comment.author}, on "
                            f"{comment.date}: "
                            f"{comment.content}")
                result.append(comment_line)
        
        return '\n'.join(result)

def nodecinfo_to_dict(node_cinfo):
    """
    将NodeCInfo实例转换为字典格式
    
    Args:
        node_cinfo: NodeCInfo实例
    
    Returns:
        dict: 转换后的字典
    """
    if not node_cinfo:
        return {}
    
    # 基本信息字典
    result = {
        # 从artInfo中提取基本信息
        'number': node_cinfo.artInfo.number,
        'title': node_cinfo.artInfo.title,
        'author': node_cinfo.artInfo.author,
        'created_at': node_cinfo.artInfo.created_at,
        'body': node_cinfo.artInfo.body,
        
        # 其他字段
        'status': node_cinfo.status,
        'milestone': node_cinfo.milestone,
        'labels': node_cinfo.labels,
        
        # 评论相关信息
        'comments_count': len(node_cinfo.comments),
        'comments': [
            {
                'author': comment.author,
                'date': comment.date,
                'content': comment.content
            }
            for comment in node_cinfo.comments
        ],
        
        # 额外的派生信息
        'latest_comment_author': node_cinfo.get_latest_comment().author if node_cinfo.get_latest_comment() else None,
        'latest_comment_date': node_cinfo.get_latest_comment().date if node_cinfo.get_latest_comment() else None,
        
        # 完整的文本表示（用于后续分析）
        'full_text': str(node_cinfo)
    }
    
    return result