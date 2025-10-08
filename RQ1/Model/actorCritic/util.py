from Utils.rl_util import generate_context_from_nodeList, preprocess_batch_graph_data, preprocess_graph_data

def get_subgraph_features(merged_result, subgraph_id):
    """从合并的结果中提取特定子图的节点特征"""
    subgraph_info = merged_result['subgraph_info'][subgraph_id]
    start_idx = subgraph_info['start_idx']
    end_idx = subgraph_info['end_idx']
    
    # 提取对应的节点特征
    subgraph_features = merged_result['node_features'][start_idx:end_idx+1]
    
    return {
        'features': subgraph_features,
        'original_nodes': subgraph_info['original_nodes'],
        'global_indices': subgraph_info['global_node_indices']
    }


def get_subgraph_edges(merged_result, subgraph_id):
    """获取特定子图在全局图中的边"""
    subgraph_info = merged_result['subgraph_info'][subgraph_id]
    global_indices = set(subgraph_info['global_node_indices'])
    
    subgraph_edges = []
    edge_index = merged_result['edge_index']
    
    # 找出属于该子图的边
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0][i].item(), edge_index[1][i].item()
        if src in global_indices and dst in global_indices:
            subgraph_edges.append([src, dst])
    
    return subgraph_edges


def get_all_subgraph_mappings(merged_result):
    """获取所有子图的映射信息摘要"""
    mappings = []
    for subgraph_info in merged_result['subgraph_info']:
        mappings.append({
            'subgraph_id': subgraph_info['subgraph_id'],
            'original_nodes': subgraph_info['original_nodes'],
            'node_count': subgraph_info['node_count'],
            'global_range': (subgraph_info['start_idx'], subgraph_info['end_idx'])
        })
    return mappings


import numpy as np
def pad_trajectories_to_numpy(trajectories, pad_value=0.0, dtype=np.float32):
    """
    将不等长的轨迹列表补齐并转换为numpy数组
    
    Args:
        trajectories: 不等长的二维列表
        pad_value: 补齐值，默认为0.0
        dtype: numpy数组类型，默认为np.float32
    
    Returns:
        numpy数组，形状为(len(trajectories), max_length)
    """
    if not trajectories:
        return np.array([], dtype=dtype)
    
    max_length = max(len(trajectory) for trajectory in trajectories)
    padded_trajectories = []
    
    for trajectory in trajectories:
        padded = trajectory + [pad_value] * (max_length - len(trajectory))
        padded_trajectories.append(padded)
    
    return np.array(padded_trajectories, dtype=dtype)