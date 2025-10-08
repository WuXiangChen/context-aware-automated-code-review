import re
import pandas as pd

def extract_validation_accuracy(log_file_path):
    """从log文件提取验证准确率并保存为Excel"""
    # 读取log文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()
    
    # 正则匹配 "Validation Accuracy at step 数字: 数字.数字"
    matches = re.findall(r'Validation Accuracy at step (\d+): ([\d.]+)', log_text)
    
    # 转换为数字并创建DataFrame
    data = [(int(step), float(acc)) for step, acc in matches]
    df = pd.DataFrame(data, columns=['Step', 'Accuracy'])
    
    # 保存为Excel
    file_path = log_file_path.replace(".log", ".xlsx")
    df.to_excel(file_path, index=False)
    return df

# 使用示例
df = extract_validation_accuracy('Output/logs/codereviewer_msg_MetaContext_finetune.log')
print(df)