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

    def get_neighbor_nodes(self, node_neighbors, cur_nodes):
        """
        基于node_neighbors字典获取当前节点的所有邻居节点
        
        Args:
            node_neighbors (dict): 邻居节点字典 {node_id: [neighbor_ids]}
            cur_nodes (list): 当前节点列表
            
        Returns:
            set: 所有邻居节点的集合
        """
        neighbor_nodes = set()
        for node_id in cur_nodes:
            if node_id in node_neighbors:
                neighbor_nodes.update(node_neighbors[node_id])
        return neighbor_nodes

    # 对应的forward方法也需要更新调用方式
    def forward(self, batch_data, current_nodes_all, node_neighbors, temperature=1.0):
        x = self.gnn_layers[0](batch_data.x, batch_data.edge_index)
        x = F.relu(x)
        x = self.gnn_layers[1](x, batch_data.edge_index)
        
        results = []
        for graph_idx, cur_nodes in enumerate(current_nodes_all):
            if len(x) not in cur_nodes:
                contexts = x[cur_nodes].mean(dim=0)
                # 限制分数范围
                node_scores = torch.tanh(self.action_head(x)).squeeze(-1)
                stop_score = torch.tanh(self.stop_head(contexts))
                # 创建mask
                mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
                # 使用新的get_neighbor_nodes方法
                neighbor_nodes = self.get_neighbor_nodes(node_neighbors, cur_nodes)
                if neighbor_nodes:
                    mask[list(neighbor_nodes)] = True
                masked_scores = node_scores.clone()
                masked_scores[~mask] = float('-inf')
                # 标准化分数
                all_scores = torch.cat([masked_scores, stop_score]) / temperature
                action_probs = F.softmax(all_scores, dim=-1)
            else:
                action_probs = torch.tensor([0.0]*len(x) + [1.0], dtype=torch.float32)
            results.append(action_probs)
        return results
        
    def sample_action(self, batch_data, current_nodes_all, node_neighbors, maxRL):
        """批量采样动作，每个子图独立采样"""
        action_probs_list = self.forward(batch_data, current_nodes_all, node_neighbors)
        
        actions, log_probs, entropies, selected_nodes, stops = [], [], [], [], []
        
        for graph_idx, action_probs in enumerate(action_probs_list):
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            selected_node = action.item()
            is_stop = (selected_node == len(batch_data.batch)) or (len(current_nodes_all[graph_idx]) > maxRL)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            selected_nodes.append(selected_node)
            stops.append(is_stop)
        
        return actions, log_probs, entropies, selected_nodes, stops