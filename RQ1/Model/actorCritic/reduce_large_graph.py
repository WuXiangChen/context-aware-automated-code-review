from collections import deque, defaultdict
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

from Database.node import GraphData


def filter_graph_data(graph_data: GraphData, selected_nodes: Set[Any]) -> GraphData:
    """根据选中的节点过滤图数据"""
    # 过滤节点
    filtered_nodes = [node for node in graph_data["nodes"] if node in selected_nodes]
    
    # 过滤边：只保留两个端点都在选中节点中的边
    filtered_edges = []
    for edge in graph_data["edges"]:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source, target = edge[0], edge[1]
            if source in selected_nodes and target in selected_nodes:
                filtered_edges.append(edge)
    
    # 过滤节点属性
    filtered_node_attrs = {
        node: attrs for node, attrs in graph_data["node_attributes"].items() 
        if node in selected_nodes
    }
    
    # 过滤边属性（如果边属性的key与边对应）
    filtered_edge_attrs = {}
    for edge in filtered_edges:
        edge_key = "-".join(edge)
        covnert_edge_key = "-".join(edge[::-1])
        if edge_key in graph_data["edge_attributes"]:
            filtered_edge_attrs[edge_key] = graph_data["edge_attributes"][edge_key]
        elif covnert_edge_key in graph_data["edge_attributes"]:
            filtered_edge_attrs[covnert_edge_key] = graph_data["edge_attributes"][covnert_edge_key]
    
    return {
        "nodes":filtered_nodes,
        "edges":filtered_edges,
        "node_attributes":filtered_node_attrs,
        "edge_attributes":filtered_edge_attrs
    }

def reduce_large_graph(graph_data: GraphData, cur_core_id: Any, max_nodes: int = 100) -> GraphData:
    """
    先筛选有属性的节点，然后在这些节点中找包含cur_core_id的连通子图
    """
    if cur_core_id not in graph_data["nodes"]:
        raise ValueError(f"core id {cur_core_id} not in graph_data")
    
    # 第一步：筛选出所有有属性的节点
    nodes_with_attrs = {
        node for node in graph_data["nodes"] 
        if node in graph_data["node_attributes"] and graph_data["node_attributes"][node]
    }
    
    # 确保cur_core_id被包含（即使它没有属性）
    nodes_with_attrs.add(cur_core_id)
    
    # 第二步：在有属性的节点中构建邻接表
    adj_list_filtered = build_adjacency_list_filtered(graph_data, nodes_with_attrs)
    
    # 第三步：从cur_core_id开始BFS，找到连通的子图
    connected_nodes = bfs_in_filtered_graph(cur_core_id, adj_list_filtered, max_nodes)
    
    return filter_graph_data(graph_data, connected_nodes)

def build_adjacency_list_filtered(graph_data: GraphData, valid_nodes: Set[Any]) -> Dict[Any, Set[Any]]:
    """在指定节点集合中构建邻接表"""
    adj_list = defaultdict(set)
    
    for edge in graph_data["edges"]:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source, target = edge[0], edge[1]
            # 只有当两个节点都在valid_nodes中时，才添加这条边
            if source in valid_nodes and target in valid_nodes:
                adj_list[source].add(target)
                adj_list[target].add(source)
    
    return adj_list

def bfs_in_filtered_graph(start_node: Any, adj_list: Dict[Any, Set[Any]], max_nodes: int) -> Set[Any]:
    """在过滤后的图中进行BFS"""
    visited = {start_node}
    queue = deque([start_node])
    
    while queue and len(visited) < max_nodes:
        current_node = queue.popleft()
        
        # 获取未访问的邻居节点
        neighbors = list(adj_list[current_node] - visited)
        
        for neighbor in neighbors:
            if len(visited) >= max_nodes:
                break
            visited.add(neighbor)
            queue.append(neighbor)
    
    return visited