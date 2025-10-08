# 本节的主要目的是，将输入的数据类型通过NoContext的设置，将其转化成符合要求的输入数据结构
from datetime import datetime
from .allocBase import allocBase
from Database.node import CodeReviewItem, GraphData

# 这里的主要目标是将所有的基类中预定义的成员属性在NoContext的上下文设置下填满
class allocLimit1Context(allocBase):
    def __init__(self, acrItem:CodeReviewItem = None, limit_n:int = 1):
        super().__init__() # 这里继承的父类中所有需要进行赋值的参数
        if acrItem is None:
            raise "The Input Data is not None!"
        
        # 其实在这里有一些内容已经是可以进行赋值了的，例如REF数据库中的全部数据
        self.codediff = acrItem.old_hunk
        self.oldf = acrItem.oldf
        self.old = acrItem.old
        self.new = acrItem.new
        self.newhunk = acrItem.hunk
        self.comment = acrItem.comment
        self.repo_id = acrItem.repo_id
        self.limit_n = limit_n
        # 这里主要是按序设计Context的
        self.initialize_attributes(acrItem.repo_id, acrItem.graph_data)
    
    # 这里将其直接邻居节点作为context
    def initialize_attributes(self, repo_id:str="", graph_data:GraphData = None):
        if not repo_id or not graph_data:
            raise f"The kernel id {repo_id} or target graph {graph_data} is empty!"
        repo_id = "PR-"+repo_id.split(":")[-1]
        # 遍历所有边，找到与repo_id相连的节点
        nodeId2CreatedAt = {}
        for key, nodeAttr in graph_data.node_attributes.items():
          if "created_at" in nodeAttr.keys():
            creatAt = nodeAttr["created_at"]
            time_str_clean = creatAt.replace('Z', '+00:00')
            dt = datetime.fromisoformat(time_str_clean)
            nodeId2CreatedAt[key] = dt
            
        latest_n_keys = [key for key, _ in sorted(nodeId2CreatedAt.items(), key=lambda x: x[1], reverse=True)[:self.limit_n+1]] + [repo_id]
        
        # 去重
        all_nodes = list(set(latest_n_keys))
        context_parts = []
        
        for node_id in all_nodes:
            if node_id in graph_data.node_attributes:
                nodeMeta = graph_data.node_attributes[node_id]
                if len(nodeMeta)!=0:
                    title = nodeMeta.get("title", "")
                    body = nodeMeta.get("body", "")
                    comments = "\n".join([con["content"] for con in nodeMeta.get("comments", "")])
                    context_parts.append(title + "\n" + body + "\n" + comments)
        
        context = "\n\n".join(context_parts)
        self.context = context