# =============================================================================
# config/settings.py
# =============================================================================

import os
from dataclasses import dataclass
from typing import List
# from dotenv import load_dotenv

@dataclass
class DatabaseConfig:
    """数据库配置"""
    connection_string: str
    database_name: str
    
    # 类方法：从环境变量加载配置, 这里得将.env放在根路径
    @classmethod
    def from_local_env(cls):
        return cls(
            connection_string=os.getenv('MONGODB_CONNECTION_STRING', 'XXX'),
            database_name=os.getenv('MONGODB_DATABASE_NAME', 'XXX'))
    
    # @classmethod
    # def from_remote_env(cls):
    #     return cls(
    #         connection_string=os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://root:wangliang123456.@172.16.66.170:17017,172.16.66.171:17117,172.16.66.172:17217/?replicaSet=mongo_cluster'),
    #         database_name=os.getenv('MONGODB_DATABASE_NAME', 'PRIM-ACR'))
    
    @classmethod
    def from_remote_env(cls):
        return cls(
            connection_string=os.getenv('MONGODB_CONNECTION_STRING', 'XXXX'),
            database_name=os.getenv('MONGODB_DATABASE_NAME', 'XXXX'))


# Linux/Mac 将下面的信息放置在启动.sh文件中
# export MONGODB_CONNECTION_STRING="mongodb://username:password@host:port"
# export MONGODB_DATABASE_NAME="production_db"

@dataclass
class AppConfig:
    """应用配置"""
    database: DatabaseConfig
    batch_size: int = 1000
    log_level: str = 'INFO'
    event_collections: List[str] = None
    
    @classmethod
    def default(cls):
        return cls(
            database=DatabaseConfig.from_local_env(),
            event_collections=[
                'IssueCommentEvent',
                'PullRequestReviewCommentEvent',
                'BaseProjectInfoGraphQL'
            ]
        )

    @classmethod
    def basicInfo(cls):
        return cls(
            database=DatabaseConfig.from_remote_env(),
            event_collections=[
                'BaseProjectInfoGraphQL',
            ]
        )
        
# 示例用法
'''
  config = AppConfig.default()
  print(config.database.connection_string)  # 输出: mongodb://localhost:27017
  print(config.batch_size)                  # 输出: 1000
'''