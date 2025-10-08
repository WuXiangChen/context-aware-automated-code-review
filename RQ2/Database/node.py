from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd


@dataclass
class GraphStatistics:
    """图统计信息"""
    num_nodes: int = field(metadata={'description': 'Number of nodes (#N)'})
    num_edges: int = field(metadata={'description': 'Number of edges (#E)'})
    num_connected_components: int = field(metadata={'description': 'Number of connected components (#CC)'})
    avg_clustering_coefficient: float = field(metadata={'description': 'Average clustering coefficient (AvgCC)'})
    clustering_coefficient: float = field(metadata={'description': 'Clustering coefficient (ClustC)'})
    avg_degree: float = field(metadata={'description': 'Average degree (AvgD)'})
    degree_centrality_entropy: float = field(metadata={'description': 'Degree centrality entropy (DegCE)'})
    batch_number: int = field(metadata={'description': 'Batch number'})


@dataclass
class GraphSummary:
    """图摘要信息"""
    num_nodes: int
    num_edges: int
    is_connected: bool


@dataclass
class GraphData:
    """图数据结构"""
    nodes: List[Any] = field(default_factory=list)
    edges: List[Any] = field(default_factory=list)
    node_attributes: Dict[str, Any] = field(default_factory=dict)
    edge_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeReviewItem:
    """代码审查项目数据类"""
    # 基本标识信息
    repo: str = field(metadata={'description': 'Repository name'})
    repo_id: str = field(metadata={'description': 'Repository ID with PR number'})
    ghid: int = field(metadata={'description': 'GitHub issue/PR ID'})
    ids: List[Union[str, int]] = field(default_factory=list, metadata={'description': 'Various IDs'})
    
    # 代码变更信息
    old_hunk: str = field(default="", metadata={'description': 'Old code hunk with context'})
    oldf: str = field(default="", metadata={'description': 'Complete old file content'})
    hunk: str = field(default="", metadata={'description': 'New code hunk with context'})
    old: str = field(default="", metadata={'description': 'Old code section'})
    new: str = field(default="", metadata={'description': 'New code section'})
    
    # 审查信息
    comment: str = field(default="", metadata={'description': 'Review comment'})
    lang: str = field(default="", metadata={'description': 'Programming language'})
    type: str = field(default="", metadata={'description': 'Type of change (e.g., test)'})
    
    # 图相关数据
    node_id: str = field(default="", metadata={'description': 'Graph node identifier'})
    graph_data: Optional[GraphData] = field(default=None, metadata={'description': 'Graph structure data'})
    statistics: Optional[GraphStatistics] = field(default=None, metadata={'description': 'Graph statistics'})
    graph_summary: Optional[GraphSummary] = field(default=None, metadata={'description': 'Graph summary'})
    timestamp: Optional[pd.Timestamp] = field(default=None, metadata={'description': 'Processing timestamp'})
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeReviewItem':
        """从字典创建 CodeReviewItem 实例"""
        # 处理图统计数据
        statistics = None
        if 'statistics' in data:
            stats_data = data['statistics']
            statistics = GraphStatistics(
                num_nodes=stats_data.get('#N', 0),
                num_edges=stats_data.get('#E', 0),
                num_connected_components=stats_data.get('#CC', 0),
                avg_clustering_coefficient=stats_data.get('AvgCC', 0.0),
                clustering_coefficient=stats_data.get('ClustC', 0.0),
                avg_degree=stats_data.get('AvgD', 0.0),
                degree_centrality_entropy=stats_data.get('DegCE', 0.0),
                batch_number=stats_data.get('batch_number', 0)
            )
        
        # 处理图摘要数据
        graph_summary = None
        if 'graph_summary' in data:
            summary_data = data['graph_summary']
            graph_summary = GraphSummary(
                num_nodes=summary_data.get('num_nodes', 0),
                num_edges=summary_data.get('num_edges', 0),
                is_connected=summary_data.get('is_connected', False)
            )
        
        # 处理图数据
        graph_data = None
        if 'graph_data' in data:
            gd = data['graph_data']
            graph_data = GraphData(
                nodes=gd.get('nodes', []),
                edges=gd.get('edges', []),
                node_attributes=gd.get('node_attributes', {}),
                edge_attributes=gd.get('edge_attributes', {})
            )
        
        return cls(
            repo=data.get('repo', ''),
            repo_id=data.get('repo_id', ''),
            ghid=data.get('ghid', 0),
            ids=data.get('ids', []),
            old_hunk=data.get('old_hunk', ''),
            oldf=data.get('oldf', ''),
            hunk=data.get('hunk', ''),
            old=data.get('old', ''),
            new=data.get('new', ''),
            comment=data.get('comment', ''),
            lang=data.get('lang', ''),
            type=data.get('type', ''),
            node_id=data.get('node_id', ''),
            graph_data=graph_data,
            statistics=statistics,
            graph_summary=graph_summary,
            timestamp=data.get('timestamp')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            'repo': self.repo,
            'repo_id': self.repo_id,
            'ghid': self.ghid,
            'ids': self.ids,
            'old_hunk': self.old_hunk,
            'oldf': self.oldf,
            'hunk': self.hunk,
            'old': self.old,
            'new': self.new,
            'comment': self.comment,
            'lang': self.lang,
            'type': self.type,
            'node_id': self.node_id,
            'timestamp': self.timestamp
        }
        
        if self.graph_data:
            result['graph_data'] = {
                'nodes': self.graph_data.nodes,
                'edges': self.graph_data.edges,
                'node_attributes': self.graph_data.node_attributes,
                'edge_attributes': self.graph_data.edge_attributes
            }
        
        if self.statistics:
            result['statistics'] = {
                '#N': self.statistics.num_nodes,
                '#E': self.statistics.num_edges,
                '#CC': self.statistics.num_connected_components,
                'AvgCC': self.statistics.avg_clustering_coefficient,
                'ClustC': self.statistics.clustering_coefficient,
                'AvgD': self.statistics.avg_degree,
                'DegCE': self.statistics.degree_centrality_entropy,
                'batch_number': self.statistics.batch_number
            }
        
        if self.graph_summary:
            result['graph_summary'] = {
                'num_nodes': self.graph_summary.num_nodes,
                'num_edges': self.graph_summary.num_edges,
                'is_connected': self.graph_summary.is_connected
            }
        
        return result


# # 使用示例
# if __name__ == "__main__":
#     # 示例数据
#     sample_data = {
#         'old_hunk': '@@ -273,6 +273,10 @@ class RootPathHandler(BaseTaskHistoryHandler):\n     def get(self...aliser/index.html")\n \n+    def head(self):',
#         'oldf': '# -*- coding: utf-8 -*-\n#\n# Copyright 2012-2015 Spotify AB\n#\n# Licensed under the Apa...()\n\n\nif __name__ == "__main__":\n    run()\n',
#         'hunk': '@@ -274,6 +274,7 @@ class RootPathHandler(BaseTaskHistoryHandler):\n         self.redi....set_status(204)\n         self.finish()\n \n',
#         'comment': 'Is the name "head" a convention for health checking? Regardless it caught me by surpr...why it exist? It should also say what 204.',
#         'ids': [19459, 'f9d15209195d2cb49052b3662bbe21e387d1a0e3', '03f712cffab169e7b617425c22eb84d82c8f081c'],
#         'repo': 'spotify/luigi',
#         'ghid': 2789,
#         'old': '         self.redirect("/static/visualiser/index.html")\n     def head(self):\n        ...elf.set_status(204)\n         self.finish()',
#         'new': '         self.redirect("/static/visualiser/index.html")\n     def head(self):\n+       ...elf.set_status(204)\n         self.finish()',
#         'lang': 'py',
#         'type': 'test',
#         'repo_id': 'spotify/luigi:2789',
#         'node_id': 'spotify/luigi:2789',
#         'graph_data': {
#             'nodes': [],
#             'edges': [],
#             'node_attributes': {},
#             'edge_attributes': {}
#         },
#         'statistics': {
#             '#N': 2,
#             '#E': 1,
#             '#CC': 1,
#             'AvgCC': 0.0,
#             'ClustC': 0.0,
#             'AvgD': 1.0,
#             'DegCE': -0.0,
#             'batch_number': 7
#         },
#         'graph_summary': {
#             'num_nodes': 2,
#             'num_edges': 1,
#             'is_connected': True
#         }
#     }
    
#     # 创建实例
#     review_item = CodeReviewItem.from_dict(sample_data)
#     print("Created CodeReviewItem:")
#     print(f"Repo: {review_item.repo}")
#     print(f"Comment: {review_item.comment[:50]}...")
#     print(f"Language: {review_item.lang}")
#     print(f"Graph nodes: {review_item.statistics.num_nodes if review_item.statistics else 'N/A'}")
    
#     # 转换回字典
#     converted_back = review_item.to_dict()
#     print(f"\nConverted back to dict keys: {list(converted_back.keys())}")