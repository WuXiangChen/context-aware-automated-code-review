# =============================================================================
# database/repositories.py
# =============================================================================

from typing import List, Dict, Any, Iterator, Optional
from datetime import datetime
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError
from models.edge import EdgeInfo
from database.connector import MongoDBConnector
from utils.logger import setup_logger

logger = setup_logger("Output/logs/"+__name__)


class EdgeRepository:
    """边数据访问层"""
    
    def __init__(self, db_connector: MongoDBConnector):
        self.db_connector = db_connector
        self.collection_name = 'edges'  # MongoDB集合名称
    
    def save_edge(self, edge: EdgeInfo) -> bool:
        """保存单条边信息"""
        try:
            collection = self.db_connector.get_collection(self.collection_name)
            collection.insert_one(edge.to_dict())
            return True
        except Exception as e:
            logger.error(f"保存边信息失败: {e}")
            return False
    
    def save_edges_batch(self, edges: List[EdgeInfo]) -> int:
        """批量保存边信息"""
        if not edges:
            return 0

        collection = self.db_connector.get_collection(self.collection_name)
        edge_docs = [edge.to_dict() for edge in edges]
        saved_count = 0  # 记录成功插入的数量

        try:
            result = collection.insert_many(edge_docs, ordered=False)
            saved_count = len(result.inserted_ids)
            logger.info(f"批量保存边信息成功: 已保存 {saved_count}/{len(edge_docs)} 条")
            return saved_count

        except BulkWriteError as bwe:
            # 部分插入失败时，BulkWriteError 仍包含成功插入的文档
            saved_count = bwe.details["nInserted"]
            logger.error(
                f"批量保存边信息部分失败: 已保存 {saved_count}/{len(edge_docs)} 条"
            )
            return saved_count

        except Exception as e:
            logger.error(f"批量保存边信息失败: {e}，已保存 {saved_count}/{len(edge_docs)} 条")
            return saved_count  # 可能是 0（完全失败）或部分成功
    
    def find_edges_by_project(self, project_name: str) -> Iterator[EdgeInfo]:
        """根据项目名查找边"""
        collection = self.db_connector.get_collection(self.collection_name)
        for doc in collection.find({'project_name': project_name}):
            yield EdgeInfo.from_dict(doc)
    
    # 这个功能基本上是不会用的
    def find_context_edges(self, target_nodes: List[str], latest_time: datetime) -> List[Dict[str, Any]]:
        """查找上下文边"""
        collection = self.db_connector.get_collection(self.collection_name)
        query = {
            '$or': [
                {'source_node_id': {'$in': target_nodes}},
                {'target_node_id': {'$in': target_nodes}}
            ],
            'edge_created_at': {'$lte': latest_time}
        }
        return list(collection.find(query))
    
    # 这个倒是可能是经常要使用的
    def edge_exists(self, source_id: str, target_id: str, project_name: str) -> bool:
        """检查边是否已存在"""
        collection = self.db_connector.get_collection(self.collection_name)
        return collection.find_one({
            'source_node_id': source_id,
            'target_node_id': target_id,
            'project_name': project_name
        }) is not None


class EventRepository:
    """事件数据访问层"""
    
    def __init__(self, basic_connector: MongoDBConnector, local_connector: MongoDBConnector = None):
        self.basic_connector = basic_connector
        self.basic_connector.connect()
        self.local_connector = local_connector
        self.local_connector.connect()
        
    # 先按照项目名称获取给定Event的所有数据
    def get_project_events(self, project_name: str, collection_name: str) -> Iterator[Dict[str, Any]]:
        """获取项目的事件数据"""
        if "BaseProjectInfoGraphQL" ==  collection_name:
            collection = self.basic_connector.get_collection(collection_name)
            project_name = "/".join(project_name.split('/')[::-1])
            query = {'repo/owner': project_name}
            for event in collection.find(query):
                yield event
        else:
            collection = self.local_connector.get_collection(collection_name)
            query = {'repo.name': project_name}
            for event in collection.find(query):
                yield event
    
    # 给定项目名称和制品节点，找到制品的创建时间
    def get_pr_created_at(self, project_name: str, pr_number: str) -> Optional[datetime]:
        """获取指定PR的创建时间"""
        collection = self.basic_connector.get_collection('BaseProjectInfoGraphQL')
        project_name = "/".join(project_name.split('/')[::-1])
        event = collection.find_one({
            'repo/owner': project_name,
            'number': int(pr_number)
        })
        return event["createdAt"] if event else None

    def get_project_node_numbers(self, project_name: str) -> Dict[str, List[str]]:
        """获取项目中所有的PR和Issue编号"""
        node_numbers = {'PR': set(), 'Issue': set()}
        # 获取PR编号
        basic_collection = self.basic_connector.get_collection('BaseProjectInfoGraphQL')
        project_name = "/".join(project_name.split('/')[::-1])
        for event in basic_collection.find({'repo/owner': project_name, 'type':"pull_request"}):
            node_numbers['PR'].add(str(event['number']))
        
        # 获取Issue编号
        for event in basic_collection.find({'repo/owner': project_name, 'type':"issue"}):
            node_numbers['Issue'].add(str(event['number']))
        
        return node_numbers