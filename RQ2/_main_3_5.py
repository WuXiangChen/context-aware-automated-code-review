# 本节的主要是为了一次性的跑完前置实验中的_3 - _5文件，方便端到端的完成实验的整理与汇总

'''
  导包区
'''
from _3_query_WithContextInfo_DS_LLM import _3_main
from _4_get_metric_for_dsLLM_result import _4_main
from _5_analysis_for_reAndContextState import _5_main
from utils.CONSTANT import BASE_MODEL


if __name__ == "__main__":
    # This script processes a dataset using a specified model and dataset name.
    for index in range(1,4,1):
      context_pickle_path =  "Data/nodeList_size_3000.pkl"
      qa_path = "Data/case_study.xlsx"
      all_model_names = BASE_MODEL.keys()
      for name in all_model_names:
        for contextFlag in [False, True]: #False, True, True, True, False
          print(f"ModelName:{name}, contextFlag:{contextFlag}")
          model_name = name
          withContext = contextFlag
          num_processes = 1
          re_path = _3_main(model_name=model_name, withContext=withContext, context_pickle_path=context_pickle_path, qa_path=qa_path, index=index, num_processes=num_processes)
          print(f"Results saved to: {re_path}")
          output_path, mean_bleu = _4_main(re_path=re_path)
          print(f"Mean value:{mean_bleu}")
          print(f"Metrics saved to: {output_path}")
          # analysis_path = _5_main(filename=output_path, original_dataset_path=original_dataset_path)
          # print(f"Analysis results saved to: {analysis_path}")