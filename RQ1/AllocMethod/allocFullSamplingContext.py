# 本节的主要目的是，将输入的数据类型通过NoContext的设置，将其转化成符合要求的输入数据结构
from .allocBase import allocBase
from Database.node import CodeReviewItem, GraphData
from typing import List, Set, Dict, Any
from collections import deque
import itertools
import random

class InstanceFullSamplingContext(allocBase):
    def __init__(self, acrItem:CodeReviewItem = None, context:str = ""):
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
        self.initialize_attributes(context=context)
        
    def initialize_attributes(self, context:str=""):
        """子类必须实现此方法来完成context属性初始化"""
        self.context = context
    
    
class allocFullSamplingContext():
    def __init__(self, acrItem:CodeReviewItem = None):
        subgraphs_with_context = self.initialize_attributes(acrItem.repo_id, acrItem.graph_data)
        self.fsContextItems = []
        for context in subgraphs_with_context:
            ifsc_ins = InstanceFullSamplingContext(acrItem=acrItem, context=context)
            self.fsContextItems.append(ifsc_ins)
    
    # RL的每一步负责将目标节点加入到上下文集合中，并将其转换成标准的context输入
    def initialize_attributes(self, repo_id:str="", graph_data:GraphData = None):
        repo_id = "PR-"+repo_id.split(":")[-1]
        core_induction_subgraphs = self.get_weakly_connected_core_subgraphs(graph=graph_data, core_node=repo_id, max_additional_nodes=10)
        # 均匀采样最多5个子图
        if len(core_induction_subgraphs) <= 5:
            sampled_subgraphs = core_induction_subgraphs
        else:
            sampled_subgraphs = random.sample(core_induction_subgraphs, 5)
        # 3. 为每个子图生成元数据
        subgraphs_with_context = []
        for subgraph in sampled_subgraphs:
            metadata = self.generate_subgraph_metadata(subgraph)
            subgraphs_with_context.append(metadata)
        return subgraphs_with_context
        
    def generate_subgraph_metadata(self, subgraph: GraphData) -> Dict[str, str]:
        """为子图中的每个节点生成元数据文本"""
        metadata = ""
        for node_id in subgraph.nodes:
            nodeMeta = subgraph.node_attributes.get(node_id, {})
            context_parts = []
            
            if nodeMeta:  # 如果有元数据
                title = nodeMeta.get("title", "").strip()
                body = nodeMeta.get("body", "").strip()
                comments = "\n".join(
                    con["content"].strip() 
                    for con in nodeMeta.get("comments", []) 
                    if con and "content" in con
                )
                
                if title: context_parts.append(title)
                if body: context_parts.append(body)
                if comments: context_parts.append(comments)
            
            metadata += "\n".join(context_parts)
        
        return metadata
    
    def get_weakly_connected_core_subgraphs(
        self,
        graph: GraphData,
        core_node: Any,
        max_additional_nodes: int = 10
    ) -> List[GraphData]:
        """
        获取所有包含指定核心节点的弱连通诱导子图（保证子图自身弱连通）
        
        参数:
            graph: 输入有向图
            core_node: 必须包含的唯一核心节点
            max_additional_nodes: 允许的最大额外节点数
        
        返回:
            满足条件的子图列表，按节点数从小到大排序
        """
        if core_node not in graph.nodes:
            return []

        # 步骤1：找到包含核心节点的弱连通分量
        component = self.find_containing_component(graph, core_node)
        if not component:
            return []

        # 步骤2：生成候选子图并验证连通性
        other_nodes = component - {core_node}
        if max_additional_nodes is not None:
            other_nodes = set(itertools.islice(other_nodes, max_additional_nodes))
        
        subgraphs = []
        for k in range(0, len(other_nodes) + 1):
            for nodes in itertools.combinations(other_nodes, k):
                node_set = {core_node}.union(nodes)
                subgraph = self.create_induced_subgraph(graph, node_set)
                
                # 验证子图是否弱连通
                if self.is_weakly_connected(subgraph):
                    subgraphs.append(subgraph)
        
        return sorted(subgraphs, key=lambda g: len(g.nodes))

    def is_weakly_connected(self, graph: GraphData) -> bool:
        """检查图是否弱连通"""
        if not graph.nodes:
            return False
        
        visited = set()
        queue = deque([graph.nodes[0]])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # 获取双向邻居
            neighbors = set()
            for (u, v) in graph.edges:
                if u == current:
                    neighbors.add(v)
                if v == current:
                    neighbors.add(u)
            
            queue.extend(n for n in neighbors if n not in visited)
        
        return len(visited) == len(graph.nodes)

    def find_containing_component(self, graph: GraphData, core_node: Any) -> Set[Any]:
        """BFS找到包含核心节点的弱连通分量"""
        visited = set()
        queue = deque([core_node])
        component = set()
        
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            
            # 获取双向邻居
            neighbors = set()
            for (u, v) in graph.edges:
                if u == current:
                    neighbors.add(v)
                if v == current:
                    neighbors.add(u)
            
            queue.extend(n for n in neighbors if n not in visited)
        
        return component

    def create_induced_subgraph(self, graph: GraphData, nodes: Set[Any]) -> GraphData:
        """创建纯净的诱导子图"""
        return GraphData(
            nodes=sorted(nodes),  # 保持节点有序
            edges=[(u, v) for (u, v) in graph.edges if u in nodes and v in nodes],
            node_attributes={n: graph.node_attributes[n] for n in nodes if n in graph.node_attributes},
            edge_attributes={(u, v): graph.edge_attributes[(u, v)] 
                            for (u, v) in graph.edges 
                            if (u, v) in graph.edge_attributes and u in nodes and v in nodes}
        )