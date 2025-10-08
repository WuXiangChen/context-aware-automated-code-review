import json
import os
import pdb
import queue
import ujson
import pickle
import random
from pathlib import Path
from typing import List, Tuple
from Database.node import CodeReviewItem
from tqdm import tqdm
import logging

logger = logging.getLogger("Output/logs/"+__name__)
class JSONLReader:
  def __init__(self, folder_path):
    file_path = None
    for fname in os.listdir(folder_path):
      if fname.endswith("valid.jsonl"):
        file_path = os.path.join(folder_path, fname)
        break
    if file_path is None:
      raise FileNotFoundError("No file ending with 'valid.jsonl' found in the folder.")
    self.file_path = file_path

  def read_lines(self, start=None, end=-1):
    """
    Read lines from the JSONL file and return them as a list of dictionaries.

    :param limit: Number of dictionaries to read. If None, read all lines.
    :return: List of dictionaries, each representing a JSON object from the file.
    """
    data = None
    try:
      with open(self.file_path, 'r', encoding='utf-8') as f:
        data = [ujson.loads(line) for line in f]
    except FileNotFoundError:
      print(f"Error: File not found at {self.file_path}")
    except json.JSONDecodeError as e:
      print(f"Error decoding JSON: {e}")
    if start is not None:
      return data[start:end]
    return data

  def write_lines(self, data):
    try:
      with open(self.file_path, 'w', encoding='utf-8') as file:
        for item in data:
          file.write(json.dumps(item, ensure_ascii=False) + '\n')
    except Exception as e:
      print(f"Error writing to file: {e}")

  def filter_lines(self, condition):
    """
    Filter lines in the JSONL file based on a condition.

    :param condition: A function that takes a dictionary and returns True if it should be included.
    :return: List of dictionaries that satisfy the condition.
    """
    data = self.read_lines()
    return [item for item in data if condition(item)]

  def count_lines(self):
    """
    Count the number of lines (JSON objects) in the JSONL file.

    :return: Number of lines in the file.
    """
    return len(self.read_lines())
  
def save_results(results_queue, model_name, dataset_name, output_dir="Results"):
    """
    Save results from the queue to a JSON file incrementally.
    Uses a consistent filename and appends new results to the existing file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate consistent filename (without timestamp)
    filename = f"{model_name}_{dataset_name}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Collect all results from the queue
    results = []
    while True:
        try:
            item = results_queue.get_nowait()
            if item is None:  # Our signal to stop
                break
            results.append(item)
        except queue.Empty:
            break
    
    # Append to JSON file if we have results
    if results:
        existing_data = []
        # Read existing data if file exists
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read existing file ({e}), starting fresh")
        
        # Combine old and new data
        combined_data = existing_data + results
        
        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nAdded {len(results)} results to {filepath} (now {len(combined_data)} total)")
    return len(results)


def split_and_save_dataset(
    data: List[CodeReviewItem],
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    train_file: str = "train_dataset.pkl",
    test_file: str = "test_dataset.pkl",
    random_seed: int = 42,
    shuffle: bool = True,
    output_dir: str = "./Data/"
) -> Tuple[int, int]:
    
    # 输入验证
    if not data:
        raise ValueError("输入数据列表不能为空")
    
    if abs(train_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"训练集和测试集比例之和必须为1.0，当前为{train_ratio + test_ratio}")
    
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("训练集和测试集比例必须大于0")
    
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 打乱数据
    if shuffle:
        random.shuffle(data)
    
    # 计算划分点
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    
    # 划分数据集
    train_dataset = data[:train_size]
    test_dataset = data[train_size:]
    
    # 构建完整文件路径
    train_path = output_path / train_file
    test_path = output_path / test_file
    
    try:
        # 保存训练集
        with open(train_path, 'wb') as f:
            pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 保存测试集
        with open(test_path, 'wb') as f:
            pickle.dump(test_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        # 日志记录
        logging.info(f"数据集划分完成:")
        logging.info(f"  总数据量: {total_size}")
        logging.info(f"  训练集: {len(train_dataset)} 条 ({len(train_dataset)/total_size*100:.1f}%)")
        logging.info(f"  测试集: {len(test_dataset)} 条 ({len(test_dataset)/total_size*100:.1f}%)")
        logging.info(f"  训练集保存至: {train_path}")
        logging.info(f"  测试集保存至: {test_path}")
        
        print(f"✅ 数据集划分并保存成功!")
        print(f"📊 训练集: {len(train_dataset)} 条数据 -> {train_path}")
        print(f"📊 测试集: {len(test_dataset)} 条数据 -> {test_path}")
        
        return len(train_dataset), len(test_dataset)
        
    except Exception as e:
        raise IOError(f"保存文件时出错: {str(e)}")


def load_dataset(file_path: str) -> List[CodeReviewItem]:
    """
    从PKL文件加载数据集
    
    Args:
        file_path: PKL文件路径
        
    Returns:
        List[CodeReviewItem]: 加载的数据集
    """
    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"✅ 成功加载数据集: {file_path}")
        print(f"📊 数据量: {len(dataset)} 条")
        
        return dataset
        
    except Exception as e:
        raise IOError(f"加载文件 {file_path} 时出错: {str(e)}")


def get_or_load_ref_nodes(data_connector, cache_path="/root/workspace/Context_Aware_ACR_Model/Data/ref_nodes.pkl", load_file=""):
    """
    获取代码评审参考节点，带有缓存功能
    如果缓存文件存在则直接加载，否则查询数据库并保存缓存
    
    Args:
        data_connector: 数据库连接器对象
        cache_path: 缓存文件路径，默认为'ref_nodes.pkl'
    
    Returns:
        List[CodeReview]: 代码评审节点列表
    """
    ref_node_list = None
    cache_file = Path(load_file)
    if cache_file.exists():
        ref_node_list = load_dataset(load_file)
        return ref_node_list
    
    # 缓存不存在或加载失败，从数据库查询
    if ref_node_list is None:
      cache_file = Path(cache_path)
      print("Querying ref nodes from database...")
      graph_base_path = "/root/workspace/Context_Aware_ACR_Model/Data/merged.pkl" # 这个东西是一定存在的！
      cache_file = Path("/root/workspace/Context_Aware_ACR_Model/Data/ref_nodes.pkl") # 保存对象
      if not cache_file.exists():
        with open(graph_base_path, 'rb') as f:
            graph_dict = pickle.load(f) # 这里加载的graph数据一定是Database/nodeInfoForContextGen.py文件中定义的NodeCInfo
        # 这里只获取所有的REF数据
        ref_node_list = data_connector.get_only_CodeReviewerDatasetREF(graph_dict)
        del graph_dict
        # 然后将获取而得的REF数据与这里的Data/merged.pkl数据进行拼接
        
        # 保存查询结果到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(ref_node_list, f)
            print(f"Saved ref nodes to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache. Error: {e}")

        ### 去除所有的无前驱节点
        ref_node_list_filtered = [item for item in ref_node_list if item.graph_summary.num_nodes!=0]
        split_and_save_dataset(ref_node_list_filtered)
        return load_dataset(load_file)

import gc
import sys
from pympler import asizeof, muppy, summary
from AllocMethod import *
def get_node_list_with_cache(args, data_connector):
    """加载或生成带缓存的node_list"""
    # 缓存文件路径
    cache_file = Path(f"/root/workspace/Context_Aware_ACR_Model/Data/alloc{args.allocMethod}Context_NodeList_{'Training' if args.train_eval else 'Test'}.pkl")
    
    # 尝试加载缓存
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                node_list = pickle.load(f)
            logger.info(f"从缓存加载 {len(node_list)} 个节点")
            if args.allocMethod == "FullSampling15":
                node_list = [item for node in node_list for item in node.fsContextItems ]
            return node_list
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}，重新生成")
    
    # 生成新数据
    logger.info("生成新的node_list...")
    if args.train_eval:
        load_file = "/root/workspace/Context_Aware_ACR_Model/Data/train_dataset.pkl"
    else:        
        load_file = "/root/workspace/Context_Aware_ACR_Model/Data/test_dataset.pkl"
    
    ref_node_list = get_or_load_ref_nodes(data_connector, load_file=load_file)
    context_func = eval(f"alloc{args.allocMethod}Context")
    
    if not args.allocMethod.startswith("FullSampling"):
        node_list = [context_func(acrItem=node) for node in tqdm(ref_node_list, desc="Processing nodes")]
    else:
        node_list = [
            item 
            for node in tqdm(ref_node_list, desc="Processing FullSampling nodes") 
            for item in context_func(acrItem=node).fsContextItems  # 假设fsContextItems是可迭代对象
        ]
    # 保存缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(node_list, f)
    
    # 清理内存
    data_connector.disconnect()
    del ref_node_list
    
    total_size = asizeof.asizeof(node_list) / (1024 ** 3)
    collected = gc.collect()
    logger.info(f"生成完成: {len(node_list)} 个节点, {total_size:.2f} GB, 回收 {collected} 个对象")
    
    return node_list
