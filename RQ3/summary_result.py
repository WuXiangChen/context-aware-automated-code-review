import pandas as pd
import glob
import os

# 指定目录（当前目录用 "."）
data_dir = "."

# 找到所有以 codediff.xlsx 结尾的文件
file_list = glob.glob(os.path.join(data_dir, "*codediff.xlsx"))

# 每块的行数
chunk_size = 10660

# 指标列
metrics = ["rouge_1", "rouge_2", "rouge_l", "precision", "recall", "perfect_pred", "bleu"]

# 汇总所有文件的 BLEU 变化
all_files_bleu_trend = []

for file_path in file_list:
    print(f"正在处理 {file_path} ...")
    
    # 读取文件
    df = pd.read_excel(file_path)
    
    # 给每一行打上块编号和块内行号
    df = df.reset_index(drop=True)
    df["block_id"] = df.index // chunk_size
    df["row_in_block"] = df.index % chunk_size

    # 逐步累积前 N 块，计算均值（只到前 20 块）
    summary_list = []
    for n in range(1, 21):
        sub_df = df[df["block_id"] < n]
        # 按行号分组，取 bleu 最大的行
        idx = sub_df.groupby("row_in_block")["bleu"].idxmax()
        result = sub_df.loc[idx].reset_index(drop=True)
        mean_values = result[metrics].mean()
        mean_values["num_blocks"] = n
        summary_list.append(mean_values)
        
        # 保存每个文件的 BLEU 变化
        all_files_bleu_trend.append({
            "file_name": os.path.basename(file_path),
            "num_blocks": n,
            "mean_bleu": mean_values["bleu"]
        })

    # 转为 DataFrame
    summary_df = pd.DataFrame(summary_list)

    # 保存每个文件的均值结果
    saved_filename = f"{os.path.splitext(file_path)[0]}_retest20_summary.xlsx"
    summary_df.to_excel(saved_filename, index=False)
    
    print(f"完成！均值结果已保存到 {saved_filename}")

# 保存所有文件的 BLEU 变化汇总
bleu_trend_df = pd.DataFrame(all_files_bleu_trend)
bleu_trend_filename = os.path.join(data_dir, "all_files_bleu_trend.xlsx")
bleu_trend_df.to_excel(bleu_trend_filename, index=False)
print(f"完成！所有文件的 BLEU 变化已保存到 {bleu_trend_filename}")
