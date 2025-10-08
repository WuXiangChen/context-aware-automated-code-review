import logging
import pickle
from tqdm import tqdm
from utils.generalLLMs.remote_server import DeepSeekClient
from utils.general_LLMs_hooker import QAProcessor
import pandas as pd
from utils.logger import setup_logger
from utils.multi_processor import multi_processor_main
from Database import *
logger = setup_logger("logs/"+__name__)


def _3_main(model_name="ds_671B", withContext=True,
            context_pickle_path="Data/all_pre_experimental_nodeCInfo_dataset_DirectNeighbor.pickle", 
            qa_path="Data/rq2_rl_train_NN_5.xlsx",
            index=0, num_processes=1):
    # This script processes a dataset using a specified model and dataset name.
    with open(context_pickle_path, 'rb') as f:
            context_infos = pickle.load(f)
    print(f"Loaded context info from {context_pickle_path}")
    df_qa = pd.read_excel(qa_path)[["repo_id", "selected_nodeLs"]]
    
    # 这里需要搞定context_info，它应当是一个字典，该字典中包含proj_id和它的context对象
    nodeToItsContext = {}   
    codediff = []
    groundTruth = []

    for _, row in df_qa.iterrows():
        proj_id = row["repo_id"]
        selectedNodeLs = eval(row["selected_nodeLs"])
        
        # 找到对应的codeCinfo
        codeCinfo = None
        for info in context_infos:
            if info.repo_id == proj_id:
                codeCinfo = info
                break
        
        if codeCinfo is None:
            continue
            
        nodeInfos = []
        allNodeAtrribute = codeCinfo.graph_data.node_attributes
        for nodeId in selectedNodeLs:
            if "full_text" in allNodeAtrribute[nodeId]:
                nodeInfos.append(allNodeAtrribute[nodeId]["full_text"])
        
        context = "/n".join(nodeInfos)
        if withContext:
            nodeToItsContext[proj_id] = context
        else:
            nodeToItsContext[proj_id] = None
        
        codediff.append(codeCinfo.old_hunk)
        groundTruth.append(codeCinfo.comment)
        
    codediff = codediff[0].strip('+# time complexity = O(n^2)')
    df_qa["codediff"] = codediff
    df_qa["groundTruth"] = groundTruth
    if withContext:
        suffix_name = "with"
    else:
        suffix_name = "No"
    return multi_processor_main(
        model_name=model_name,
        context_info=nodeToItsContext,
        df_qa=df_qa,
        suffix_name=suffix_name,
        index=index,
        num_processes=num_processes)
