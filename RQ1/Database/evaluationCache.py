import hashlib
import json
import os
from typing import Dict, Any, Optional

class EvaluationCache:
    def __init__(self, cache_file: str = "evaluation_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, float] = {}
        self.load_cache()
    
    def load_cache(self):
        """从文件加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached evaluations")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
            msg = f"Cache successfully saved to {self.cache_file}"
            print(msg)  # Print to console
        except Exception as e:
            msg = f"Unexpected error saving cache: {e}"
            print(f"Error: {msg}")  # Print to console
            raise
    
    def get(self, sha_key: str) -> Optional[float]:
        """获取缓存的评估结果"""
        return self.cache.get(sha_key)
    
    def set(self, sha_key: str, reward: float):
        """设置缓存的评估结果"""
        self.cache[sha_key] = reward
        # 可选：每次更新后自动保存
        # self.save_cache()
    
    def exists(self, sha_key: str) -> bool:
        """检查是否存在缓存"""
        return sha_key in self.cache
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)