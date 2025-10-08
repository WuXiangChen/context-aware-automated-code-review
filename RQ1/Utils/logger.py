"""
日志配置模块
"""
import logging
import os
from datetime import datetime

def setup_logger(log_file_path: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_file_path: 日志文件路径，如果为None则使用默认路径
    
    Returns:
        配置好的logger对象
    """
    # 如果没有指定路径，创建默认的日志文件路径
    if log_file_path is None:
        # 创建logs目录（如果不存在）
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用时间戳创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"github_scraper_{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger('github_scraper')
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter - 增加详细的错误行信息
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    
    # 设置formatter
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger