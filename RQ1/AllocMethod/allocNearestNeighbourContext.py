# 本节的主要目的是，将输入的数据类型通过NoContext的设置，将其转化成符合要求的输入数据结构
from .allocBase import allocBase
from Database.node import CodeReviewItem, GraphData

# 这里的主要目标是将所有的基类中预定义的成员属性在NoContext的上下文设置下填满
class allocNearestNeighbourContext(allocBase):
    def __init__(self, acrItem:CodeReviewItem = None):
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
        # 这里主要是按序设计Context的
        self.initialize_attributes(acrItem.repo_id, acrItem.graph_data)
    
    # 这里将其直接邻居节点作为context
    def initialize_attributes(self, repo_id:str="", graph_data:GraphData = None):
        if not repo_id or not graph_data:
            raise f"The kernel id {repo_id} or target graph {graph_data} is empty!"
    
        neighbors = []
        repo_id = "PR-"+repo_id.split(":")[-1]
        # 遍历所有边，找到与repo_id相连的节点
        for edge in graph_data.edges:
            if edge[0] == repo_id:
                neighbors.append(edge[1])
            elif edge[1] == repo_id:
                neighbors.append(edge[0])
        
        # 去重
        neighbors = list(set(neighbors))

        # 构建上下文：核心节点 + 所有邻居节点的title和body
        all_nodes = [repo_id] + neighbors
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