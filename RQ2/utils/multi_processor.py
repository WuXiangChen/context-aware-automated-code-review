import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from utils.generalLLMs.remote_server import DeepSeekClient
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_row(args):
    """
    处理单行数据的函数，用于多进程调用
    
    Args:
        args: tuple包含 (index, row_data, context_info)
    
    Returns:
        dict: 包含处理结果的字典
    """
    index, code_diff,pr_id, pr2context, model_name = args
    try:
        ds = DeepSeekClient(base_model=model_name)
        # 执行代码审查
        result = ds.perform_code_review(code_diff=code_diff, pr_id=pr_id, additional_context=pr2context)        
        return {'index': index, 'status': 'success', 'result': result}
        
    except Exception as e:
        logger.error(f"处理索引 {index} 失败: {str(e)}")
        return {
            'index': index,
            'status': 'error',
            'result': None,
            'error': str(e)
        }

def process_dataframe_multiprocess(df_qa, model_name, context_info, num_processes=None):
    """
    使用多进程处理DataFrame
    
    Args:
        df_qa: 要处理的DataFrame
        context_info: 上下文信息字典
        num_processes: 进程数，默认为CPU核心数
    
    Returns:
        tuple: (处理后的DataFrame, 失败记录列表)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    logger.info(f"开始多进程处理，使用 {num_processes} 个进程")
    
    # 准备数据
    df_result = df_qa.copy()
    failed_records = []
    
    args_list = []
    for index, row in df_qa.iterrows():
        pr_id = row['repo_id']
        if context_info is None:
            pr2context = None
        else:
            pr2context = context_info.get(pr_id, "")
        args_list.append((index, row['codediff'], pr_id, pr2context, model_name))
    
    # 使用进程池处理
    with Pool(processes=num_processes) as pool:
        # results = list(tqdm(pool.imap_unordered(process_single_row, args_list),total=len(args_list),desc="处理进度"))
        results = list(tqdm(pool.imap_unordered(process_single_row, args_list),total=len(args_list),desc="处理进度"))
    
    # 处理结果
    for result in tqdm(results):
        index = result['index']
        if result['status'] == 'success':
            df_result.at[index, 'ds_dnContext'] = result['result']
        else:
            df_result.at[index, 'ds_dnContext'] = None
            failed_records.append({'index': index,'error': result['error']})
    
    logger.info(f"处理完成，成功: {len(df_result) - len(failed_records)}, 失败: {len(failed_records)}")
    return df_result, failed_records

# 使用示例
def multi_processor_main(df_qa, context_info, model_name="ds_671B",  suffix_name="", index=0, num_processes=1):
    # 多进程处理
    df_result, failed_records = process_dataframe_multiprocess(df_qa=df_qa, model_name=model_name, context_info=context_info, num_processes=num_processes)
    saved_name = f'Results/processed_{model_name}_{suffix_name}_Context_results_{index}.pkl'
    df_result.to_pickle(saved_name)
    # 输出失败记录
    if failed_records:
        logger.warning(f"有 {len(failed_records)} 条记录处理失败")
        for record in failed_records:
            logger.warning(f"索引 {record['index']}: {record['error']}")
    logger.info("处理完成")
    return saved_name