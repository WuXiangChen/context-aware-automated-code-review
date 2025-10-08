import logging
import multiprocessing
import os
import random
import torch
import numpy as np
import cProfile
import pstats
import signal
from Database.connector import MongoDBConnector
from Database.settings import AppConfig
from Model import *
from AllocMethod import *
from Model._1_BaseTrainer.utils import RefineDataset, build_or_load_gen_model
from Model.actorCritic.RLTrainer import SimplifiedRLTrainer
from Utils.rl_util import generate_dataloader_for_rl
from config import get_args
from Model.actorCritic.actor import ActorNetwork
from Model.actorCritic.critic import CriticNetwork
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告
logger = logging.getLogger("Output/logs/"+__name__)
def set_seed(seed=42):
    # Python 内置随机数
    random.seed(seed)
    
    # NumPy 随机数
    np.random.seed(seed)
    
    # PyTorch 随机数（CPU）
    torch.manual_seed(seed)
    
    # PyTorch 随机数（GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡
    
    # cuDNN 确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 固定 Python hash 种子（涉及一些数据加载）
    os.environ["PYTHONHASHSEED"] = str(seed)
# state定义为什么？很简单，就是已选节点集合的context embedding
# critic的输入模型是什么？很简单，就是context embedding，codediff embedding 和 对应某个state的action；这表明模型在学习的是当前(s,a)状态-动作对的Q函数
# actor的输入模型是什么？ 很简单，就是当前图的节点特征（图上每个节点的节点特征，就定义为其自然文本拼接的embedding值）以及边集合
    
def update_args_for_rl():
    """创建简化的参数配置"""
    class Args:
        def __init__(self):
            # 训练参数
            self.actor_lr = 1e-3
            self.critic_lr = 1e-3
            self.entropy_weight = 0.01
            self.max_grad_norm = 1.0
            self.batch_size = 1
            self.node_feature_dim = 256
            self.codediff_dim = 256
            self.context_dim = 512
            self.action_dim = 512
            self.hidden_dim = 128
            self.num_episodes = 3
            # 日志和保存
            self.log_interval = 10
            self.save_interval = 10
            # self.cpu_count = multiprocessing.cpu_count()
            self.cpu_count = 100
            self.gamma = 0.9
            self.output_dir = f"../ContextAware_ACR_Model_Saved/rl_{predefined_args.task_type}/{predefined_args.rlModel}_{predefined_args.task_type}/"
    
    return Args()

def save_profile_stats(profiler, filename):
    """保存性能分析结果"""
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumtime')
    stats.dump_stats(filename)  # 保存二进制结果
    stats.print_stats(10)      # 打印前10耗时函数

if __name__ == "__main__":
    set_seed(42)
    predefined_args = get_args()
    args = update_args_for_rl()
    
    ##########################
    print("="*50)
    print("RL Actor-Critic Training Starting with Parameters:")
    print(f"Model Name: {predefined_args.model_type}")
    print(f"Actor LR: {args.actor_lr}, Critic LR: {args.critic_lr}")
    print(f"Episodes: {args.num_episodes}")
    print("="*50)
    print()
    ##########################
    
    actor = ActorNetwork(node_feature_dim=args.node_feature_dim, hidden_dim=args.hidden_dim)
    critic = CriticNetwork(state_dim=args.context_dim, hidden_dim=args.hidden_dim)
    if predefined_args.task_type=="msg":
        # predefined_args.model_name_or_path = f"../ContextAware_ACR_Model_Saved/codereviewer/msg/allocFullSamplingContext/checkpoint-3600-15.0000/"
        # predefined_args.model_name_or_path  = "../ContextAware_ACR_Model_Saved/contextAware/msg/allocFullSamplingContext/checkpoint-70800-11.0100/"
        # predefined_args.model_name_or_path  = "../ContextAware_ACR_Model_Saved/contextAware/msg/allocFullSamplingContext/checkpoint-4800-14.7100/"
        # predefined_args.model_name_or_path = "../ContextAware_ACR_Model_Saved/contextAware/msg/allocFullSamplingContext/checkpoint-1200-14.3500"
        predefined_args.model_name_or_path = "../ContextAware_ACR_Model_Saved/contextAware/msg/allocFullSamplingContext/checkpoint-4000-15.0000"
    elif predefined_args.task_type=="ref":
        predefined_args.model_name_or_path = f"../ContextAware_ACR_Model_Saved/codereviewer/ref/allocFullSamplingContext/checkpoint-14400-84.9000/"
    predefined_bleu_model = eval(f"{predefined_args.model_type}Model")(args=predefined_args, config=None)
    _, predefined_bleu_model, tokenizer = build_or_load_gen_model(args=predefined_args, model=predefined_bleu_model)
    pool = multiprocessing.Pool(args.cpu_count)
    rfDataTrans = RefineDataset(tokenizer=tokenizer, pool=pool, args=predefined_args, file_path=[], samplenum=-1, ref_node_list=None)
    
    # 确认保存路径存在
    os.makedirs(args.output_dir, exist_ok=True)
        
    # 这里主要是为了生成dataloader
    # data_config = AppConfig.default()
    # data_connector = MongoDBConnector(data_config.database)
    # data_connector.connect()
    data_connector = None
    train_flag = predefined_args.train_eval
    if train_flag:
        train_dataloader, valid_dataloader = generate_dataloader_for_rl(predefined_args, data_connector, batch_size=args.batch_size, train_flag=train_flag)
        trainer = SimplifiedRLTrainer(args, actor, critic, predefined_bleu_model,  rfDataTrans, pool, predefined_args)
        trainer.run(train_dataloader, valid_dataloader, num_epochs=1000)
    else:
        test_dataloader = generate_dataloader_for_rl(predefined_args, data_connector, batch_size=args.batch_size, train_flag=train_flag)
        trainer = SimplifiedRLTrainer(args, actor, critic, predefined_bleu_model,  rfDataTrans, pool, predefined_args)
        rlModel = predefined_args.rlModel.replace("test","train")
        trainer.load_models(model_path=f"../ContextAware_ACR_Model_Saved/rl_{predefined_args.task_type}/{rlModel}/rl_model_best.pt")
        mean_reward, allRe_df = trainer.evaluate(test_dataloader, "test")
        allRe_df.to_excel(f'evaluation_results_rl_{predefined_args.maxRL}_codediff.xlsx', index=False)
        print(f"mean_bleu:{mean_reward}")
        print(f"all metrics results have been save in evaluation_results_rl_NN{predefined_args.maxRL}.xlsx")

    print("&"*50)
    print(f"Final Model Name: {predefined_args.model_type}")
    print(f"Final Dataset Name: {predefined_args.dataset_name}")
    print("RL Actor-Critic Training Completed!")