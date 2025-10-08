import json
import json
import os
import queue
import pandas as pd
from  tqdm import tqdm
import ujson

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


def pr_quintuple_to_string(pr2context):
    """
    将PR上下文信息转换为字符串格式
    
    Args:
        pr2context: 包含'5-tuple'和'node_attribute'的字典
    
    Returns:
        str: 格式化的字符串
    """
    lines = []
    if pr2context is None:
      return ""
    
    # 处理5-tuple信息
    if '5-tuple' in pr2context:
        lines.append("\n📊 5-TUPLE RELATIONSHIPS:")
        lines.append("-" * 40)
        tuples = pr2context['5-tuple']
        lines.append(f"Total tuples: {len(tuples)}")
        
        for i, tuple_data in enumerate(tuples):
            lines.append(f"\nTuple {i+1}:")
            lines.append(f"  Subject: {tuple_data.get('subject', 'N/A')}")
            lines.append(f"  Mention Type: {tuple_data.get('mention-type', 'N/A')}")
            lines.append(f"  Object: {tuple_data.get('object', 'N/A')}")
            lines.append(f"  Edge Attributes: {tuple_data.get('edge_attributes', 'N/A')}")


import networkx as nx
def save_to_mongodb(nodeId_graph_dict, df_stats, edge_connector, prInfo_connector):
    """
    将nodeId对应的图信息和统计数据保存到MongoDB
    
    Args:
        nodeId_graph_dict: 节点ID到图的字典
        df_stats: 统计信息DataFrame
        edge_connector: MongoDB边连接器
        prInfo_connector: MongoDB PR信息连接器
    """
    # 获取集合
    edge_collection = edge_connector.get_collection('nodeId2GraphAndSTAs')
    prInfo_collection = prInfo_connector.get_collection('nodeId2GraphAndSTAs')
    
    print("正在保存数据到MongoDB...")
    
    # 将df_stats转换为字典，便于查找
    stats_dict = df_stats.set_index('node_id').to_dict('index')
    
    documents_to_insert = []
    
    for node_id, graph in tqdm(nodeId_graph_dict.items(), desc="Preparing MongoDB documents"):
        # 获取对应的统计信息
        stats = stats_dict.get(node_id, {})
        
        # 将NetworkX图转换为可序列化的格式
        graph_data = {
            'nodes': list(graph.nodes()),
            'edges': list(graph.edges()),
            'node_attributes': {str(node): graph.nodes[node] for node in graph.nodes()},
            'edge_attributes': {f"{str(u)}-{str(v)}": graph.edges[u, v] for u, v in graph.edges()}
        }
        
        # 构建文档
        document = {
            'node_id': node_id,
            'graph_data': graph_data,
            'statistics': stats,
            'timestamp': pd.Timestamp.now(),
            'graph_summary': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'is_connected': nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
            }
        }
        
        documents_to_insert.append(document)
    
    # 批量插入到两个连接器的集合中
    from pymongo.errors import BulkWriteError

    if documents_to_insert:
        # 插入到edge_connector
        try:
            result = edge_collection.insert_many(documents_to_insert, ordered=False)
            print(f"成功向edge_connector的nodeId2GraphAndSTAs集合插入了 {len(result.inserted_ids)} 条记录")
        except BulkWriteError as bwe:
            # 获取成功插入的数量
            inserted_count = bwe.details.get('nInserted', 0)
            write_errors = bwe.details.get('writeErrors', [])
            print(f"edge_connector插入完成: 成功 {inserted_count} 条，失败 {len(write_errors)} 条")
            
            # 可选：打印部分错误信息（避免输出过多）
            if write_errors:
                print(f"edge_connector错误示例: {write_errors[0].get('errmsg', '未知错误')}")
        except Exception as e:
            print(f"向edge_connector插入数据时出现未预期错误: {e}")
        
        # 插入到prInfo_connector
        try:
            result = prInfo_collection.insert_many(documents_to_insert, ordered=False)
            print(f"成功向prInfo_connector的nodeId2GraphAndSTAs集合插入了 {len(result.inserted_ids)} 条记录")
        except BulkWriteError as bwe:
            # 获取成功插入的数量
            inserted_count = bwe.details.get('nInserted', 0)
            write_errors = bwe.details.get('writeErrors', [])
            print(f"prInfo_connector插入完成: 成功 {inserted_count} 条，失败 {len(write_errors)} 条")
            
            # 可选：打印部分错误信息
            if write_errors:
                print(f"prInfo_connector错误示例: {write_errors[0].get('errmsg', '未知错误')}")
        except Exception as e:
            print(f"向prInfo_connector插入数据时出现未预期错误: {e}")



def save_to_pkl(nodeId_graph_dict, df_stats, output_path="documents_to_insert.pkl"):
    """
    将nodeId对应的图信息和统计数据处理成documents_to_insert格式并保存为pkl文件
    
    Args:
        nodeId_graph_dict: 节点ID到图的字典
        df_stats: 统计信息DataFrame
        output_path: 输出文件路径，默认为"documents_to_insert.pkl"
    """
    import pickle
    import pandas as pd
    import networkx as nx
    from tqdm import tqdm
    
    print("正在保存数据到pkl文件...")
    
    # 将df_stats转换为字典，便于查找
    stats_dict = df_stats.set_index('node_id').to_dict('index')
    
    documents_to_insert = []
    
    for node_id, graph in tqdm(nodeId_graph_dict.items(), desc="Preparing documents"):
        # 获取对应的统计信息
        stats = stats_dict.get(node_id, {})
        
        # 将NetworkX图转换为可序列化的格式
        graph_data = {
            'nodes': list(graph.nodes()),
            'edges': list(graph.edges()),
            'node_attributes': {str(node): graph.nodes[node] for node in graph.nodes()},
            'edge_attributes': {f"{str(u)}-{str(v)}": graph.edges[u, v] for u, v in graph.edges()}
        }
        
        # 构建文档
        document = {
            'node_id': node_id,
            'graph_data': graph_data,
            'statistics': stats,
            'timestamp': pd.Timestamp.now(),
            'graph_summary': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'is_connected': nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
            }
        }
        
        documents_to_insert.append(document)
    
    # 保存documents_to_insert到pkl文件
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(documents_to_insert, f)
        print(f"成功保存 {len(documents_to_insert)} 条文档到: {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


def load_from_pkl(pkl_path="documents_to_insert.pkl"):
    """
    从pkl文件加载documents_to_insert数据
    
    Args:
        pkl_path: pkl文件路径
    
    Returns:
        list: documents_to_insert列表
    """
    import pickle
    
    try:
        with open(pkl_path, 'rb') as f:
            documents_to_insert = pickle.load(f)
        print(f"成功从 {pkl_path} 加载了 {len(documents_to_insert)} 条文档")
        return documents_to_insert
    except Exception as e:
        print(f"加载pkl文件时出错: {e}")
        return []