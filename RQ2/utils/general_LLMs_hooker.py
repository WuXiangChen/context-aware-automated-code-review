from tqdm import tqdm
from utils.data_util import JSONLReader, save_results
from utils.generalLLMs.remote_server import DeepSeekClient
import concurrent.futures
from tqdm import tqdm
import queue

class QAProcessor:
    def __init__(self, model_name, dataset_name, max_workers=1, save_interval=5):
        """
        初始化QA处理器
        
        参数:
            model_name: 使用的模型名称
            dataset_name: 数据集名称
            max_workers: 最大并发工作线程数
            save_interval: 保存结果的间隔
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.save_interval = save_interval
    
    def _process_single_qa(self, qa, cli, results_queue):
        """
        处理单个QA对
        
        参数:
            qa: 包含问题和答案的字典
            cli: 模型客户端
            results_queue: 结果队列
        """
        print(qa.keys())
        input_ = qa["patch"]
        while True:
            try:
                output = cli.perform_code_review(input_)
                qa_ = {"Question": input_, "RawOutput": str(output)}
                merged_qa = {**{k: v for k, v in qa.items() if k != "patch"}, **qa_}
                results_queue.put(merged_qa)
                break
            except Exception as e:
                print(f"Error occurred during processing: {e}. Retrying...")
                continue
    
    def _process_concurrently(self, QAs, cli):
        """
        并发处理QA对
        
        参数:
            QAs: QA对列表
            cli: 模型客户端
        """
        results_queue = queue.Queue()
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for qa in QAs:
                futures.append(executor.submit(
                    self._process_single_qa,
                    qa=qa,
                    cli=cli,
                    results_queue=results_queue))
            
            # 进度条
            for _ in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(QAs), 
                         desc="Processing QAs"):
                processed_count += 1
                
                # 定期保存
                if processed_count % self.save_interval == 0:
                    self._save_results(results_queue)
            
            # 保存剩余结果
            if processed_count % self.save_interval != 0:
                results_queue.put(None)
                self._save_results(results_queue)
    
    def _save_results(self, results_queue):
        """
        保存结果到文件

        参数:
            results_queue: 结果队列
        """

        save_results(results_queue, self.model_name, self.dataset_name)
        print(f"Results {self.model_name}_{self.dataset_name} have been saved!")
    
    def process_dataset(self, data_path, start=None, sample_size=2):
        """
        处理整个数据集
        
        参数:
            data_path: 数据文件路径
            start: 开始行(可选)
            end: 结束行(可选)
            sample_size: 如果未指定start/end，则采样大小
        """
        if data_path is None:
            raise ValueError("Please provide a valid data file path")
        
        jsReader = JSONLReader(data_path)
        count_len = jsReader.count_lines()
        
        # 确定处理范围
        end = -1
        if start is not None:
            start = start
            end = start + sample_size
        
        QAs = jsReader.read_lines(start=start, end=end)
        cli = DeepSeekClient(base_model=self.model_name)
        
        # 开始处理
        self._process_concurrently(QAs, cli)


# # 使用示例
# if __name__ == "__main__":
#     processor = QAProcessor(
#         model_name="deepseek-code",
#         dataset_name="code_review",
#         max_workers=4,
#         save_interval=10
#     )
    
#     processor.process_dataset(
#         data_path="path/to/your/data.jsonl",
#         sample_size=200
#     )