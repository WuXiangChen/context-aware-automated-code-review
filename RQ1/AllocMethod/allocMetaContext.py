# 本节的主要目的是，将输入的数据类型通过NoContext的设置，将其转化成符合要求的输入数据结构
from .allocBase import allocBase
from Database.node import CodeReviewItem, GraphData

# 这里的主要目标是将所有的基类中预定义的成员属性在NoContext的上下文设置下填满
class allocMetaContext(allocBase):
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
    
    def initialize_attributes(self, repo_id:str="", graph_data:GraphData = None):
        # 本节是NoContext配置，因此，所有的context都为""
        node_id = "PR-"+repo_id.split(":")[-1]
        nodeMeta = graph_data.node_attributes[node_id]
        context = ""
        if len(nodeMeta)!=0:
            context = nodeMeta["title"] + "\n" + nodeMeta["body"]
        self.context = context