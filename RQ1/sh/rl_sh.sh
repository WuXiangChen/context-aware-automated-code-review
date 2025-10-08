# 训练

nohup python -u rl_main.py --rlModel msg/train_NN_10 --device "cuda:0" --llm_device "cuda:1" --task_type "msg" --maxRL 10 > ./Output/rl_logs/msg/run_train_NN10.log 2>&1 &

nohup python -u rl_main.py --rlModel ref/train_NN_10_baseBatchIndex_0 --device "cuda:0" --llm_device "cuda:1" --task_type "ref" --maxRL 10 --baseBatchIndex 0 > ./Output/rl_logs/ref/run_train_NN10_baseBatchIndex_0.log 2>&1 &
nohup python -u rl_main.py --rlModel ref/train_NN_10_baseBatchIndex_11 --device "cuda:2" --llm_device "cuda:3" --task_type "ref" --maxRL 10 --baseBatchIndex 11 > ./Output/rl_logs/ref/run_train_NN10_baseBatchIndex_11.log 2>&1 &

nohup python -u rl_main.py --rlModel msg/train_NN_10_baseBatchIndex_0 --device "cuda:4" --llm_device "cuda:5" --task_type "msg" --maxRL 10 --baseBatchIndex 0 > ./Output/rl_logs/msg/run_train_NN10_baseBatchIndex_0.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_10_baseBatchIndex_11 --device "cuda:6" --llm_device "cuda:7" --task_type "msg" --maxRL 10 --baseBatchIndex 11 > ./Output/rl_logs/msg/run_train_NN10_baseBatchIndex_11.log 2>&1 &

# 测试
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:0" --llm_device "cuda:0" --maxRL 1 > run_contextAware_codediff_maxRL1_1_1.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:1" --llm_device "cuda:1" --maxRL 2 > run_contextAware_codediff_maxRL2_1_1.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:2" --llm_device "cuda:2" --maxRL 3 > run_contextAware_codediff_maxRL3_1_1.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:3" --llm_device "cuda:3" --maxRL 4 > run_contextAware_codediff_maxRL4_1_1.log 2>&1 &

nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:1" --llm_device "cuda:1" --maxRL 5 > run_contextAware_codediff_maxRL5_1_1.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:2" --llm_device "cuda:2" --maxRL 6 > run_contextAware_codediff_maxRL6_1_2.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:3" --llm_device "cuda:3" --maxRL 7 > run_contextAware_codediff_maxRL7_1_2.log 2>&1 &
nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:0" --llm_device "cuda:0" --maxRL 8 > run_contextAware_codediff_maxRL8_1_1.log 2>&1 &

nohup python -u rl_main.py --rlModel msg/train_NN_5 --train_eval --task_type "msg" --device "cuda:0" --llm_device "cuda:1" --maxRL 10 > run_contextAware_codediff_maxRL10_1_1.log 2>&1 &
