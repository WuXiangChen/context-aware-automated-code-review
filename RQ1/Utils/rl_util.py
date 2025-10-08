import datetime
import time
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from typing import Any, List, Tuple
from Database.node import GraphData
from typing import Tuple, List, Dict
from Metrics.smooth_bleu import bleu_fromstr
from torch.utils.data import DataLoader, random_split
from Model._1_BaseTrainer.smooth_bleu import rouge_pp_metrics
from Utils.data_util import  get_or_load_ref_nodes
from torch.utils.data import DataLoader, Dataset

def generate_context_from_nodeList(graph_data, current_nodes):
    """为子图中的每个节点生成元数据文本"""
    metadata = ""
    for node_id in current_nodes:
        if node_id in graph_data["node_attributes"] and len(graph_data["node_attributes"].get(node_id))!=0:
            nodeMeta = graph_data["node_attributes"].get(node_id)
        else:
            convert_node_id = lambda node_id: node_id.replace("PR", "Issue", 1) if node_id.startswith("PR") else node_id.replace("Issue", "PR", 1) if node_id.startswith("Issue") else node_id
            nodeMeta = graph_data["node_attributes"].get(convert_node_id(node_id), {})
        context_parts = []
        if nodeMeta:  # 如果有元数据
            title = nodeMeta.get("title", "").strip()
            body = nodeMeta.get("body", "").strip()
            comments = "\n".join( con["content"].strip()  for con in nodeMeta.get("comments", []) if con and "content" in con)
            if title: context_parts.append(title)
            if body: context_parts.append(body)
            if comments: context_parts.append(comments)
        metadata += "\n".join(context_parts)
    return metadata

def custom_collate_fn(batch):
  crList = []
  for index, item in enumerate(batch):
    item_dict = item.to_dict()
    core_id = "PR-"+str(item_dict["ghid"])
    core_attr = item_dict["graph_data"]["node_attributes"][core_id]
    item_dict["id"] = str(index)
    context = ""
    if core_attr:
      context = core_attr["title"] + "\n" + core_attr["body"]
    item_dict["context"] = context
    item_dict["codediff"] = item_dict["old_hunk"] 
    crList.append(item_dict)
  return crList 

class NodeListDataset(Dataset):
  def __init__(self, node_list):
    self.node_list = node_list

  def __len__(self):
    return len(self.node_list)

  def __getitem__(self, idx):
    return self.node_list[idx]

# 这里需要一个单独的方法为rl_main的执行过程准备dataloader
def generate_dataloader_for_rl(args, data_connector, batch_size, train_flag):
    if train_flag:
        # 训练模式：创建训练和验证数据集
        load_file = "/root/workspace/Context_Aware_ACR_Model/Data/train_dataset_simple.pkl"
        nodeList = get_or_load_ref_nodes(data_connector, load_file=load_file)
        train_dataset = NodeListDataset(nodeList)
        
        load_file = "/root/workspace/Context_Aware_ACR_Model/Data/test_dataset_Filtered.pkl"
        nodeList = get_or_load_ref_nodes(data_connector, load_file=load_file)
        test_dataset = NodeListDataset(nodeList)
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # 训练时打乱数据
            collate_fn=custom_collate_fn
        )
        
        valid_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # 验证时不打乱数据
            collate_fn=custom_collate_fn
        )
        
        return train_dataloader, valid_dataloader
    else:
        load_file = "/root/workspace/Context_Aware_ACR_Model/Data/test_dataset_simple.pkl"
        # load_file = "./Data/nodeList_size_200.pkl"
        # load_file = "/root/workspace/Context_Aware_ACR_Model/Data/test_dataset_Filtered.pkl"
        nodeList = get_or_load_ref_nodes(data_connector, load_file=load_file)
        dataset = NodeListDataset(nodeList)
        dataloader = DataLoader(dataset, batch_size=11000, shuffle=False, collate_fn=custom_collate_fn)
        return dataloader
        
def preprocess_graph_data(
    graph_data: GraphData, 
    core_id:str = "",
    node_to_idx:dict={},
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将GraphData预处理为sample_action函数所需的输入格式（有向图，要求必须有特征）
    Args:
        graph_data: 图数据结构（必须包含节点特征）
        current_nodes: 当前已选择的节点数量
        device: 设备类型 ('cpu' 或 'cuda')
    Returns:
        node_features: 节点特征张量 [num_nodes, feature_dim]
        edge_index: 边索引张量 [2, num_edges]
        current_nodes: 当前节点数量张量
    Raises:
        ValueError: 如果图数据中没有节点特征
    """
    
    # 1. 验证节点特征存在
    if 'features' not in graph_data["node_attributes"]:
        raise ValueError("图数据中必须包含节点特征 (node_attributes['features'])")
    
    # 2. 处理节点特征
    num_nodes = len(graph_data["nodes"])
    
    if num_nodes == 0:
        # 空图情况
        node_features = torch.zeros((0, 0), dtype=torch.float32, device=device)
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        return node_features, edge_index
    
    # 获取节点特征
    features = graph_data["node_attributes"]['features']
    if isinstance(features, list):
        node_features = torch.tensor(features, dtype=torch.float32, device=device)
    else:
        node_features = torch.as_tensor(features, dtype=torch.float32, device=device)
    
    # core_id_index = [node_to_idx[_id] for _id in core_id]
    
    # 验证特征维度
    if node_features.size(0) != num_nodes:
        raise ValueError(f"节点特征数量 ({node_features.size(0)}) 与节点数量 ({num_nodes}) 不匹配")
    
    # 3. 处理边索引（有向图）
    if len(graph_data["edges"]) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        edge_list = []
        for edge in graph_data["edges"]:
            if len(edge) >= 2:
                src_node, dst_node = edge[0], edge[1]
                # if dst_node in core_id_index and src_node not in core_id_index:
                #    src_node, dst_node = edge[1], edge[0]
                edge_list.append([src_node, dst_node])
                edge_list.append([dst_node, src_node])
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    return node_features, edge_index

def evaluate_batch_examples(
    model: torch.nn.Module,
    examples: List[Any],
    tokenizer: Any,
    device: str,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    max_target_length: int = 512,
    batch_size: int = 20,
    llm_device = "cuda:1"
) -> List[float]:
    """
    批量评估多个样本的模型预测效果
    
    Args:
        model: 要评估的模型
        examples: 样本列表，每个必须包含 source_ids 和 target_ids
        tokenizer: 分词器
        device: 使用的设备
        beam_size: beam search大小
        length_penalty: 长度惩罚
        max_target_length: 最大生成长度
        batch_size: 分批大小，默认30
    
    Returns:
        bleu_scores: BLEU分数列表
    """
    model.eval()
    actual_model = model.module if hasattr(model, "module") else model
    
    total_examples = len(examples)
    device = torch.device(llm_device)
    # 如果样本数小于等于batch_size，直接处理
    if total_examples <= batch_size:
        return _process_single_batch(
            actual_model, examples, tokenizer, device, 
            beam_size, length_penalty, max_target_length
        )
    
    # 分批处理
    all_bleu_scores = []
    all_other_metrics = []
    for i in range(0, total_examples, batch_size):
        # 获取当前批次的样本
        end_idx = min(i + batch_size, total_examples)
        batch_examples = examples[i:end_idx]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_examples + batch_size - 1)//batch_size}, "
              f"samples {i+1}-{end_idx}/{total_examples}")
        
        # 处理当前批次
        batch_bleu_scores, _ = _process_single_batch(
            actual_model, batch_examples, tokenizer, device,
            beam_size, length_penalty, max_target_length
        )
        
        # 合并结果
        all_bleu_scores.extend(batch_bleu_scores)
        all_other_metrics.extend(_)
        torch.cuda.empty_cache()
        print("cuda 已经释放")
    return all_bleu_scores, all_other_metrics

def _process_single_batch(
    model: torch.nn.Module,
    examples: List[Any],
    tokenizer: Any,
    device: str,
    beam_size: int,
    length_penalty: float,
    max_target_length: int
) -> List[float]:
    """
    处理单个批次的样本
    
    Args:
        model: 要评估的模型
        examples: 当前批次的样本列表
        tokenizer: 分词器
        device: 使用的设备
        beam_size: beam search大小
        length_penalty: 长度惩罚
        max_target_length: 最大生成长度
    
    Returns:
        bleu_scores: 当前批次的BLEU分数列表
    """
    # 准备批量输入
    source_ids = torch.stack([torch.tensor(ex.source_ids, dtype=torch.long) for ex in examples]).to(device)
    target_ids_list = [ex.target_ids for ex in examples]
    model.to(device=device)
    # 批量生成
    with torch.no_grad():
        preds = model.generate(
            input_ids=source_ids,
            attention_mask=source_ids.ne(tokenizer.pad_id),
            use_cache=True,
            synced_gpus=False,
            num_beams=beam_size,
            early_stopping=True,
            length_penalty=length_penalty,
            max_length=max_target_length
        )
    
    # 处理批量结果
    pred_strs = []
    gold_strs = []
    
    for i in range(len(examples)):
        pred_str = tokenizer.decode(
            preds[i][1:],  # 去掉第一个token
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        gold_str = tokenizer.decode(
            target_ids_list[i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        pred_strs.append(pred_str)
        gold_strs.append(gold_str)
    
    # 批量计算BLEU分数
    bleu_scores, _ = bleu_fromstr_batch(pred_strs, gold_strs, rmstop=False)
    
    return bleu_scores, _

def bleu_fromstr_batch(pred_strs: List[str], gold_strs: List[str], rmstop: bool = False) -> List[float]:
    """
    批量计算BLEU分数
    
    Args:
        pred_strs: 预测字符串列表
        gold_strs: 参考字符串列表
        rmstop: 是否移除停用词
    
    Returns:
        bleu_scores: 每个样本的BLEU分数列表
    """
    bleu_scores = []
    other_metrics = []
    for pred, gold in zip(pred_strs, gold_strs):
        # 这里实现你的BLEU计算逻辑
        # 可以是调用NLTK的bleu_score或自定义实现
        score = bleu_fromstr([pred], [gold], rmstop)
        other_metric = rouge_pp_metrics([pred], [gold])
        bleu_scores.append(score)
        other_metrics.append(other_metric)
    return bleu_scores, other_metrics

def get_simple_model_signature(model, beam_size, tokenizer, args):
    """
    生成简化的模型签名，只包含最关键的信息
    """
    signature_parts = []
    
    # 模型类型
    if hasattr(model, '__class__'):
        signature_parts.append(model.__class__.__name__)
    
    # 关键配置
    if hasattr(model, 'config'):
        config = model.config
        # 只选择最影响结果的参数
        key_attrs = ['vocab_size', 'hidden_size', 'num_hidden_layers']
        for attr in key_attrs:
            if hasattr(config, attr):
                signature_parts.append(f"{attr}:{getattr(config, attr)}")
    
    # Beam size
    signature_parts.append(f"beam:{beam_size}")
    
    # Tokenizer
    if hasattr(tokenizer, 'vocab_size'):
        signature_parts.append(f"tok_vocab:{tokenizer.vocab_size}")
    signature_parts.append(f"max_source_length:{args.max_source_length}")
    signature_parts.append(f"path:{args.model_name_or_path}")
    
    return "|".join(signature_parts)

def connected_components_custom(edge_index, num_nodes):
    edge_list = edge_index.t().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)
    
    components = list(nx.connected_components(G))
    comp_id = torch.empty(num_nodes, dtype=torch.long)
    for idx, comp in enumerate(components):
        for node in comp:
            comp_id[node] = idx
    return comp_id, len(components)

def preprocess_batch_graph_data(
    graph_data: dict, 
    core_id=[],
    node_to_idx: dict = {},
    device: str = 'cpu'
) -> Tuple[Batch, Dict[int, List[int]]]:
    """
    将GraphData预处理为Batch格式，每个连通子图一个Data对象
    Args:
        graph_data: 图数据结构（必须包含节点特征）
        core_id: 核心节点ID列表
        node_to_idx: 节点ID到索引的映射（可选）
        device: 'cpu' 或 'cuda'
    Returns:
        batch_data: Batch对象，包含所有连通子图
        node_neighbors: 每个节点的邻居节点字典 {node_id: [neighbor_ids]}
    """
    
    # 1. 验证节点特征存在
    if 'features' not in graph_data["node_attributes"]:
        raise ValueError("图数据中必须包含节点特征 (node_attributes['features'])")
    
    # 2. 节点特征处理
    num_nodes = len(graph_data["nodes"])
    if num_nodes == 0:
        return Batch(), {}
    
    features = graph_data["node_attributes"]['features']
    node_features = torch.as_tensor(features, dtype=torch.float32, device=device)
    if node_features.size(0) != num_nodes:
        raise ValueError(f"节点特征数量 ({node_features.size(0)}) 与节点数量 ({num_nodes}) 不匹配")
    
    # 3. 构建邻居节点字典
    node_neighbors = {i: [] for i in range(num_nodes)}
    for edge in graph_data.get("edges", []):
        if len(edge) >= 2:
            src, dst = edge[0], edge[1]
            if dst not in node_neighbors[src]:
                node_neighbors[src].append(dst)
            if src not in node_neighbors[dst]:
                node_neighbors[dst].append(src)
    
    # 对邻居列表排序
    for node_id in node_neighbors:
        node_neighbors[node_id].sort()
    
    # 4. 边索引（无向化）
    edge_list = []
    for edge in graph_data["edges"]:
        if len(edge) >= 2:
            src_node, dst_node = edge[0], edge[1]
            edge_list.append([src_node, dst_node])
            edge_list.append([dst_node, src_node])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # 5. 找连通子图
    comp_ids, num_comps = connected_components_custom(edge_index, num_nodes=num_nodes)
    
    data_list = []
    
    for comp_id in range(num_comps):
        sub_nodes = (comp_ids == comp_id).nonzero(as_tuple=True)[0]
        sub_edge_index, _ = subgraph(sub_nodes, edge_index, relabel_nodes=True)
        
        sub_features = node_features[sub_nodes]
        
        data = Data(x=sub_features, edge_index=sub_edge_index)
        data_list.append(data)
    
    # 6. 合成 Batch
    batch_data = Batch.from_data_list(data_list).to(device)
    
    return batch_data, node_neighbors