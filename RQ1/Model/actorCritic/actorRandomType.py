from numpy import dtype
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from Model import *
from AllocMethod import *
import torch.nn.functional as F


'''
    GCN特征提取: 两层图卷积网络提取节点特征，已选节点的均值作为上下文表示；
    邻接约束选择: 只能从当前已选节点集合的邻接节点中选择下一个节点，已选节点被mask掉；
    概率输出: 线性层输出所有节点的选择分数加上停止动作分数，通过softmax得到动作概率分布；
'''
from torch_geometric.nn import GATConv
class ActorNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, heads=4):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GATConv(node_feature_dim, hidden_dim, heads=heads),  # 多头注意力
            GATConv(hidden_dim * heads, hidden_dim, heads=1)     # 单头输出
        ])
        self.action_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Linear(hidden_dim, 1)
        # 初始化权重
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.xavier_uniform_(self.stop_head.weight, gain=0.01)

    def get_connected_components_scipy(self, edge_index, num_nodes):
        """
        使用scipy获取连通分量
        """
        try:
            from torch_geometric.utils import to_scipy_sparse_matrix
            from scipy.sparse.csgraph import connected_components as scipy_connected_components
            
            # 转换为scipy稀疏矩阵
            adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
            
            # 获取连通分量
            n_components, component_labels = scipy_connected_components(
                adj_matrix, directed=False, return_labels=True
            )
            
            return torch.tensor(component_labels, dtype=torch.long, device=edge_index.device)
        except ImportError as e:
            raise ImportError(f"scipy not available: {e}. Please install scipy: pip install scipy")

    def get_batch_connected_component_masks(self, edge_index, current_nodes_all, num_nodes):
        """
        批量获取所有连通分量的mask
        
        Args:
            edge_index: 边索引 [2, num_edges]
            current_nodes_all: 所有图的当前节点集合列表
            num_nodes: 总节点数
        
        Returns:
            list: 每个图对应的mask tensor
        """
        # 一次性获取所有连通分量标签
        component_labels = self.get_connected_components_scipy(edge_index, num_nodes)
        
        masks = []
        for cur_nodes in current_nodes_all:
            if not cur_nodes or num_nodes in cur_nodes:  # 空列表处理
                mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
                masks.append(mask)
                continue
            
            # 创建mask：属于相同连通分量但不是已选择的节点
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
            # 转换current_nodes为tensor
            current_nodes_tensor = torch.tensor(cur_nodes, device=edge_index.device)
            # 获取当前节点所在的连通分量标签
            current_component_labels = component_labels[current_nodes_tensor]
            
            
            
            # 向量化操作：找到所有属于相同连通分量的节点
            """检查连通分量标签的合法性"""
            assert torch.all(component_labels >= 0), "存在负的连通分量标签!"
            assert torch.all(component_labels < num_nodes), "连通分量标签超出节点范围!"
            assert component_labels.device == current_nodes_tensor.device, "设备不一致!"
            for comp_label in current_component_labels.unique():
                same_component = (component_labels == comp_label)
                mask = mask | same_component
            
            # 排除已选择的节点
            mask[current_nodes_tensor] = False
            masks.append(mask)
        
        return masks
    
    def forward(self, batch_data, current_nodes_all, node_neighbors, temperature=1.0):
        """
        超级优化版本 - 最大化向量化操作
        """
        x = self.gnn_layers[0](batch_data.x, batch_data.edge_index)
        x = F.relu(x)
        x = self.gnn_layers[1](x, batch_data.edge_index)
        
        # 一次性计算所有节点的分数
        node_scores = torch.tanh(self.action_head(x)).squeeze(-1)
        
        # 批量获取masks
        masks = self.get_batch_connected_component_masks(batch_data.edge_index, current_nodes_all, x.size(0))
        
        # 批量计算所有context和stop_scores
        contexts_list = []
        stop_conditions = []
        
        for cur_nodes in current_nodes_all:
            if cur_nodes and len(x) not in cur_nodes:
                contexts = x[cur_nodes].mean(dim=0)
                contexts_list.append(contexts)
                stop_conditions.append(False)
            else:
                contexts_list.append(torch.zeros_like(x[0]))
                stop_conditions.append(True)
        
        # 批量计算stop scores
        if contexts_list:
            contexts_batch = torch.stack(contexts_list)
            stop_scores_batch = torch.tanh(self.stop_head(contexts_batch))
        else:
            stop_scores_batch = torch.tensor([], device=x.device)
        
        # 批量生成结果
        results = []
        for graph_idx, (mask, stop_condition) in enumerate(zip(masks, stop_conditions)):
            if not stop_condition:
                # 正常情况：计算action概率
                masked_scores = node_scores.clone()
                masked_scores[~mask] = float('-inf')
                
                stop_score = stop_scores_batch[graph_idx] if len(stop_scores_batch) > graph_idx else torch.tensor(0.0, device=x.device)
                all_scores = torch.cat([masked_scores, stop_score]) / temperature
                action_probs = F.softmax(all_scores, dim=-1)
            else:
                # 停止条件
                action_probs = torch.tensor([0.0] * len(x) + [1.0], dtype=torch.float32, device=x.device)
            
            results.append(action_probs)
        
        return results
    
    def sample_action(self, batch_data, current_nodes_all, node_neighbors):
        """批量采样动作，每个子图独立采样"""
        action_probs_list = self.forward(batch_data, current_nodes_all, node_neighbors)
        actions, log_probs, entropies, selected_nodes, stops = [], [], [], [], []
        for graph_idx, action_probs in enumerate(action_probs_list):
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            selected_node = action.item()
            is_stop = (selected_node == len(batch_data.batch)) or (len(current_nodes_all[graph_idx]) > 15)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            selected_nodes.append(selected_node)
            stops.append(is_stop)
        return actions, log_probs, entropies, selected_nodes, stops