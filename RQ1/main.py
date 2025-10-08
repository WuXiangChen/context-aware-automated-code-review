import logging
import os
import gc
import sys

from Database.connector import MongoDBConnector
from Database.settings import AppConfig
from Model import *
from AllocMethod import *
from config import get_args
from Utils.data_util import get_node_list_with_cache
logger = logging.getLogger("Output/logs/"+__name__)


if __name__ == "__main__":
  args = get_args()
  ##########################
  print("="*50)
  print("Program Starting with Parameters:")
  if args.general_model:
    print(f"Model Name: {args.model_name}")
  else:
    print(f"Model Name: {args.model_type}")
  print(f"Dataset Name: {args.dataset_name}")
  print("="*50)
  print()
  ##########################

  # 做程序入口
  # 确定数据集的加载路径
  datafile_path = f"../ACR_Dataset/{args.dataset_name}/{args.task_type}"
  
  g_bool = args.general_model
  result_output_folder = f"Results/{args.dataset_name}/{args.task_type}"
  os.makedirs(result_output_folder, exist_ok=True)
  
  data_config = AppConfig.default()
  data_connector = MongoDBConnector(data_config.database)
  data_connector.connect()
  
  node_list = get_node_list_with_cache(args, data_connector)
  logger.info(f"nodeclass has been loaded from ref_node_list, its length is {len(node_list)}")
  
  # 分配模型与任务类型
  configPath = f"./Model/{args.model_type}"
  config = os.path.join(configPath, "config.json")
  args.model_name_or_path = f"../ACR_Model_Saved/{args.model_type}/originalModel/" # 这里集中了模型信息加载的基本内容，包括config、base_model
  model = eval(f"{args.model_type}Model")(args=args, config=None)
  args.output_dir = f"../ContextAware_ACR_Model_Saved/{args.model_type}/{args.task_type}/alloc{args.allocMethod}Context/"
  os.makedirs(args.output_dir, exist_ok=True)
  
  # 将模型注入到训练过程中
  if args.train_eval:
    logger.info("Start Training")
    logger.info(f"Training/eval parameters: model_name_or_path={args.model_name_or_path}, output_dir={args.output_dir}")
    trainer = eval(f"{args.model_type}{args.task_type.upper()}")(args=args, data_file=[], model=model, eval_=False, ref_node_list=node_list)
    del node_list
    trainer.run()

  else:
    # 测试
    logger.info("Start Testing")
    logger.info(f"Testing/eval parameters: model_name_or_path={args.model_name_or_path}, output_dir={args.output_dir}")
    tester = eval(f"{args.model_type}{args.task_type.upper()}")(args=args, data_file=[], model=model, eval_=True, ref_node_list=node_list)
    test_dataloader = tester.get_data_loader(train_eval_=True)
    del node_list
    metrics = tester.evaluate(test_dataloader)
    print("test_metrics:",metrics)


  # Create Results directory if it doesn't exist
  print("&"*50)
  print(f"Final Model Name: {args.model_type}")
  print(f"Final Dataset Name: {args.dataset_name}")