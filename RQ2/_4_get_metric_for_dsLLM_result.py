# 本节的主要作用是，提取 pre_experiment_Results_directNeighborsContextInfo 中的实验结果，并进行计算指标与统计分析

'''
导包区
'''
import json
from venv import logger
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
from Metrics.smooth_bleu import bleu_fromstr
from rouge import Rouge
# from bert_score import score

def rouge_pp_metrics(predictions, eval_examples):
    """计算评估指标"""
    rouge_1, rouge_2, rouge_l, perfect_pred, precision, recall = 0.0, 0.0, 0.0, 0, 0.0, .0
    rouge = Rouge()
    
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(eval_examples, str):
        eval_examples = [eval_examples]
    
    for inf, gold in zip(predictions, eval_examples):
        # 计算完美预测
        if inf.strip().lower() == gold.strip().lower():
            perfect_pred += 1
        
        # 计算BLEU-4
        # if inf.strip() == "":
        #     dev_bleu += 0.0
        # else:
        #     dev_bleu += sentence_bleu([gold.strip().split()], inf.strip().split())
        
        # 计算ROUGE
        if inf.strip() == "":
            scores = rouge.get_scores(" ", gold.strip())
        else:
            scores = rouge.get_scores(inf.strip().lower(), gold.strip().lower())
        
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        precision += scores[0]['rouge-1']['p']
        recall += scores[0]['rouge-1']['r']
    
    total = len(predictions)
    return {
        'rouge_1': rouge_1 / total,
        'rouge_2': rouge_2 / total,
        'rouge_l': rouge_l / total,
        'precision': precision / total,
        'recall': recall / total,
        'perfect_pred': perfect_pred / total}


def _4_main(re_path="Results/processed_ds_671B_pre_experiment_limit35_Context_results_0.pkl"):
    """
    Process experiment results, compute metrics (BLEU, ROUGE), and save them to an Excel file.
    """
    # 读取实验结果
    with open(re_path, 'rb') as f:
        pre_re_directNeighbors = pickle.load(f)
    df_pre_re_directNeighbors = pd.DataFrame(pre_re_directNeighbors)
    
    # 计算指标
    results = []
    ## 逐个样本评估指标
    for index, item in tqdm(df_pre_re_directNeighbors.iterrows(), total=len(df_pre_re_directNeighbors)):
        golds = [item['groundTruth']]
        try:
            ds_re = json.loads(item["ds_dnContext"])
        except Exception as e:
            ds_re = {"code_review_suggestion": item["ds_dnContext"]}
        
        try:
            pred_nls = [ds_re["code_review_suggestion"]]
            metrics = rouge_pp_metrics(pred_nls, golds)
            metrics['bleu'] = bleu_fromstr(pred_nls, golds, rmstop=False)
        except Exception as e:
            print(f"ROUGE metric计算失败: {str(e)}, {type(pred_nls), type(golds)}")
            # 计算失败时设置所有指标为0
            metrics = {
                'bleu': 0,
                'rouge_1': 0,
                'rouge_2': 0,
                'rouge_l': 0,
                'precision': 0,
                'recall': 0,
                'perfect_pred': 0
            }
        
        result = {
            'pr_id': item['repo_id'],
            'bleu': metrics['bleu'],
            'rouge_1': metrics['rouge_1'],
            'rouge_2': metrics['rouge_2'],
            'rouge_l': metrics['rouge_l'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'perfect_pred': metrics['perfect_pred']
        }
        results.append(result)

    # 保存成xlsx
    df = pd.DataFrame(results)
    mean_bleu = df["bleu"].mean()
    suffix_name = re_path.replace('.pkl', '').replace('processed_', '').replace("Results/","")
    output_path = f"Results_Metrics/re_{suffix_name}.xlsx"
    df.to_excel(output_path, index=False)
    return output_path, mean_bleu