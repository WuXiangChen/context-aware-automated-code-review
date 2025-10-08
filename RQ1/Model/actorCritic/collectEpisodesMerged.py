import pickle
import hashlib
import os
from pathlib import Path
import pandas as pd
from Model.actorCritic.reduce_large_graph import reduce_large_graph
from Utils.rl_util import generate_context_from_nodeList, preprocess_batch_graph_data

def collect_episodes_merged(crItems, batch_idx, rfDataTrans, predefined_args, device, core_ids, test_flag="train", cache_dir="/root/workspace/Context_Aware_ACR_Model/Data/"):
    """将多个离散的图整合成一张大图，并记录每个子图的初始节点
    
    Args:
        crItems: 图项目列表
        rfDataTrans: 数据转换器
        predefined_args: 预定义参数
        device: 设备
        core_ids: 核心节点ID列表
        cache_dir: 缓存目录路径
    
    Returns:
        dict: 包含batch_data, node_neighbors, global_node_to_idx, core_ids, crItems的字典
    """
    
    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 生成缓存键（基于输入数据的哈希值）
    # cache_key = _generate_cache_key(crItems, predefined_args, core_ids)
    if test_flag=="test": test_flag="valid"
    cache_file = cache_path / f"merged_graph_batch_idx_{batch_idx}_{test_flag}.pkl"
    
    # 检查缓存是否存在
    if cache_file.exists():
        print(f"Found cached result, loading from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            
            # 验证缓存数据完整性
            if _validate_cached_result(cached_result):
                print("Cache loaded successfully!")
                return cached_result
            else:
                print("Cache validation failed, regenerating...")

        except Exception as e:
            raise f"Error loading cache: {e}, regenerating..."
    
    print("Generating new merged graph...")
    
    # 原始的图合并逻辑
    # 初始化合并后的图数据结构
    merged_graph_data = {
        "nodes": [],
        "edges": [],
        "node_attributes": {"features": []},
        "edge_attributes": {}
    }
    
    # 记录子图信息
    subgraph_info = []
    core_id_index = []
    global_node_to_idx = {}
    global_idx = 0
    crItems_ = []
    
    # 遍历每个crItem，整合图数据
    for subgraph_id, crItem in enumerate(crItems):
        cur_core_id = core_ids[subgraph_id]
        graph_data = crItem["graph_data"]
        if len(graph_data["nodes"]) > 100: # 假设存储在crItem中
            if cur_core_id is not None:
                # 缩减图数据
                graph_data = reduce_large_graph(graph_data, cur_core_id, max_nodes=100)
                # print(f"子图 {subgraph_id}: 从 {len(graph_data["nodes"])} 节点缩减到 {len(reduced_graph["nodes"])} 节点")
                # graph_data = reduced_graph
            else:
                print(f"子图 {subgraph_id}: 找不到 cur_core_id，跳过处理")
                continue
        # 记录当前子图的起始节点索引
        subgraph_start_idx = global_idx
        subgraph_nodes = []
        subgraph_original_nodes = graph_data["nodes"].copy()  # 保存原始节点列表

        crItems_.append(crItem)
        flag_core = True
        # 处理节点：重新编号并添加到全局图中
        local_to_global_mapping = {}
        for local_idx, node_id in enumerate(graph_data["nodes"]):
            # 创建全局唯一的节点ID
            global_node_id = f"subgraph_{subgraph_id}_node_{node_id}"
            merged_graph_data["nodes"].append(global_node_id)
            if node_id in core_ids and flag_core:
                core_id_index.append(global_idx)
                flag_core = False
            
            # 建立映射关系
            local_to_global_mapping[node_id] = global_idx
            global_node_to_idx[global_node_id] = global_idx
            subgraph_nodes.append(global_idx)
            
            global_idx += 1
        
        # 处理边：更新边的节点索引
        for edge in graph_data.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                src_node, dst_node = edge
                global_src = local_to_global_mapping[src_node]
                global_dst = local_to_global_mapping[dst_node]
                merged_graph_data["edges"].append([global_src, global_dst])
        
        # 处理节点特征
        for node_id in graph_data["nodes"]:
            node_context = generate_context_from_nodeList(graph_data, [node_id])
            feature = rfDataTrans.encode_remove(rfDataTrans.tokenizer, node_context, args=predefined_args)
            feature, *_ = rfDataTrans.pad_assert(feature, [], args=predefined_args, tokenizer=rfDataTrans.tokenizer)
            merged_graph_data["node_attributes"]["features"].append(feature)
        
        # 记录子图信息
        subgraph_info.append({
            'subgraph_id': subgraph_id,
            'original_nodes': subgraph_original_nodes,  # 原始节点ID列表
            'global_node_indices': subgraph_nodes,      # 在合并图中的全局索引
            'local_to_global_mapping': local_to_global_mapping,  # 本地到全局的映射
            'node_count': len(subgraph_nodes),
            'start_idx': subgraph_start_idx,
            'end_idx': global_idx - 1
        })
    
    core_id_index = list(core_id_index)
    
    # 处理合并后的图数据
    batch_data, node_neighbors = preprocess_batch_graph_data(
        graph_data=merged_graph_data, 
        device=device, 
        core_id=core_id_index, 
        node_to_idx=global_node_to_idx
    )
    
    # 构建最终结果
    result = {
        'batch_data': batch_data,
        'node_neighbors': node_neighbors,
        'global_node_to_idx': global_node_to_idx,
        'core_ids': core_id_index,
        "crItems": crItems_,
        'subgraph_info': subgraph_info,  # 添加子图信息到结果中
    }
    
    # 保存到缓存
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Result saved to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return result


def _validate_cached_result(cached_result):
    """验证缓存结果的完整性"""
    required_keys = ['batch_data', 'node_neighbors', 'global_node_to_idx', 'core_ids', 'crItems']
    
    if not isinstance(cached_result, dict):
        return False
    
    for key in required_keys:
        if key not in cached_result:
            return False
    
    # 检查数据是否为空
    if not cached_result['crItems']:
        return False
        
    if not isinstance(cached_result['global_node_to_idx'], dict):
        return False
        
    if not isinstance(cached_result['node_neighbors'], dict):
        return False
    
    return True