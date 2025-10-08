# 本节的主要作用是，将各种实验结果与其所在上下文的原始状态进行合并分析
from unittest import result
import pandas as pd
import os
import pickle
from tqdm import tqdm
import networkx as nx
from math import log2

def calculate_metrics(G):
      """计算所有SUMMARY指标"""
      metrics = {}
      if G.number_of_nodes() == 0:
        metrics['#N'] = 0
        metrics['#E'] = 0
        metrics['#CC'] = 0
        metrics['AvgCC'] = 0
        metrics['ClustC'] = 0
        metrics['AvgD'] = 0
        metrics['DegCE'] = 0
        return metrics
      
      # 基础指标
      metrics['#N'] = G.number_of_nodes()
      metrics['#E'] = G.number_of_edges()
      metrics['#CC'] = nx.number_connected_components(G)
      
      # 聚类相关
      metrics['AvgCC'] = nx.average_clustering(G)
      metrics['ClustC'] = nx.transitivity(G)  # 全局聚类系数
      
      # 度数相关
      degrees = dict(G.degree()).values()
      metrics['AvgD'] = sum(degrees) / metrics['#N'] if metrics['#N'] > 0 else 0
      
      # 度中心性熵
      degree_centrality = nx.degree_centrality(G).values()
      entropy = -sum(p * log2(p) for p in degree_centrality if p > 0)
      metrics['DegCE'] = entropy
      
      return metrics

import os
import pickle
import pandas as pd
from tqdm import tqdm

def _5_main(filename="", original_dataset_path="Data/pre_experimental_original_context_dataset.pickle"):
    """
    Load experimental results, enrich them with graph-based metrics, and save the updated analysis.
    """
    result_df = pd.read_excel(filename)

    if os.path.exists(original_dataset_path):
        with open(original_dataset_path, "rb") as f:
            context_info = pickle.load(f)
    
    for index, row in tqdm(result_df.iterrows(), total=result_df.shape[0]):
        repo_name = "/".join(row['repo'].split("/")[::-1])
        pr_id = int(row['pr_id'])
        
        # Get context information
        context_key = f"{repo_name}:{str(pr_id)}"
        pr2context = context_info.get(context_key, "")
        
        if pr2context == "":
            print(f"Warning: No context found for {context_key}")
            continue
        
        # Analyze graph structure and compute metrics
        metric = calculate_metrics(pr2context)
        
        # Update DataFrame with computed metrics
        for key, value in metric.items():
            result_df.at[index, key] = value
    
    # Save the enriched results
    filename = filename.split('Results_Metrics/')[-1]
    save_path = os.path.join("Analysis", filename)
    os.makedirs("Analysis", exist_ok=True)  # Ensure directory exists
    result_df.to_excel(save_path, index=False, engine='openpyxl')
    return save_path
# if __name__ == "__main__":
#     _5_main()
  

  