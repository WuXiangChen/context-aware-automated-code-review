import argparse

def get_args():
    parser = argparse.ArgumentParser(description="The entrence function for S-ACR process.")
    parser.add_argument("-g", "--general_model", type=bool, default=False, help="The general LLMs are used, rather than dedicated")
    parser.add_argument("-m", "--model_name", type=str, choices=["gpt_4o", "ds_671B", "ds_reasoner", "qwen_2.5", "llama_3.1"], default="qwen_2.5",help="The model to be used (options: gpt_4o, ds_671B, qwen_2.5, llama_3.1)")
    parser.add_argument("-d", "--dataset_name", type=str, choices=["CR", "CarLLM", "T5CR"], default="CR", help="The dataset given to be used")
    ########### 以上是为，通用模型的模型参数选择；以下是为专用模型的参数选择 ###########
    parser.add_argument("-md", "--model_type", type=str, default="codereviewer", choices=["codereviewer", "t5cr", "codefinder", "llaMa_reviewer", "codedoctor", "codet5", "inferFix", "auger","jLED", "DAC"])
    parser.add_argument("-ts", "--task_type", type=str, choices=["cls", "msg", "ref"], default="msg")
    parser.add_argument("--train_eval", action="store_false", default=True)
    parser.add_argument("--add_lang_ids", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    parser.add_argument("--raw_input", action="store_true", help="Whether to use simple input format (set for baselines).")
    parser.add_argument("--output_dir", default=None, type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, required=False)
    parser.add_argument("--length_penalty", default=1.0, type=float,help="The length penalty of model generation")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Path to trained model: Should contain the .bin files")
    # parser.add_argument("--do_train", action="store_true", help="Whether to run eval on the train set.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--train_batch_size", default=8, type=int,help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--mask_rate", default=0.15, type=float, help="The masked rate of input lines.")
    parser.add_argument("--beam_size", default=6, type=int, help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", default=-1, type=int,)
    parser.add_argument("--log_steps", default=-1, type=int,)
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--gpu_per_node",type=int,default=4,help="gpus per node")
    parser.add_argument(
        "--node_index",
        type=int,
        default=0,
        help="For distributed training: node_index",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--max_source_length",
        default=256,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=256,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--allocMethod",
        default="Limit1",
        type=str,
        choices=["No", "Meta", "NearestNeighbour", "FullSampling", "FullSampling15", "Limit1", "Limit5"],
        help="Allocation Method to generate Context Choice: No(无), Meta(元数据), NearestNeighbour(最近邻)",
    )
    
    parser.add_argument(
        "--rlModel",
        default="train_NN_5",
        type=str,
        help="RL Method to generate Context Choice: NearestNeighbour(NN, 最近邻), Random(随机选择)"
    )
    
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help=""
    )
    
    parser.add_argument(
        "--llm_device",
        default="cuda:1",
        type=str,
        help=""
    )
    
    parser.add_argument(
        "--maxRL",
        default=5,
        type=int,
        help=""
    )
    
    parser.add_argument(
        "--baseBatchIndex",
        default=10,
        type=int,
        help=""
    )
        
    parser.add_argument("--seed", type=int, default=2233, help="random seed for initialization")  # previous one 42
    args = parser.parse_args()
    return args