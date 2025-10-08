# 本节的主要目的是作为一个Interface而存在，用以规划未来各种上下文设计的标准输入输出。
from abc import ABC, abstractmethod
from typing import Any

class allocBase(ABC):
    def __init__(self):
        """初始化所有属性为None"""
        self.id = ""
        self.context = None
        self.codediff = None
        self.newhunk = None
        self.comment = None
        self.oldf = None
        self.old = None
        self.new = None
        self.repo_id = None

    # ========== 字典式访问接口 ==========
    def __getitem__(self, key: str) -> Any:
        """通过['key']方式获取属性"""
        if not hasattr(self, key):
            raise KeyError(f"Invalid attribute: {key}")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """通过['key']=value方式设置属性"""
        if not hasattr(self, key):
            raise KeyError(f"Invalid attribute: {key}")
        setattr(self, key, value)
        
    # ========== 抽象方法 ==========
    @abstractmethod
    def initialize_attributes(self):
        """子类必须实现此方法来完成context属性初始化"""
        pass
    
    # ========== 属性访问器 ==========
    @property
    def id(self):
        """获取id值"""
        return self.__dict__.get('id', "")
    
    @id.setter
    def id(self, value):
        """设置id值"""
        self.__dict__['id'] = value
    
    # ========== Context 属性访问器 ==========
    @property
    def context(self):
        """获取context值"""
        return self.__dict__.get('context')
    
    @context.setter
    def context(self, value):
        """设置context值"""
        self.__dict__['context'] = value

    # ========== CodeDiff 属性访问器 ==========
    @property
    def codediff(self):
        """获取codediff值"""
        return self.__dict__.get('codediff')
    
    @codediff.setter
    def codediff(self, value):
        """设置codediff值"""
        self.__dict__['codediff'] = value
        
    # ========== NewHunk 属性访问器 ==========
    @property
    def newhunk(self):
        """获取newhunk值"""
        return self.__dict__.get('newhunk')
    
    @newhunk.setter
    def newhunk(self, value):
        """设置newhunk值"""
        self.__dict__['newhunk'] = value

    # ========== Messages 属性访问器 ==========
    @property
    def comment(self):
        """获取messages"""
        return self.__dict__.get('comment', "")
    
    @comment.setter
    def comment(self, value):
        """设置messages"""
        if value is None:
            self.__dict__['comment'] = ""
        elif isinstance(value, (list, tuple)):
            self.__dict__['comment'] = "\n".join(str(item) for item in value)
        else:
            self.__dict__['comment'] = value

    # ========== 文件路径属性访问器 ==========
    @property
    def oldf(self):
        """获取旧文件路径"""
        return self.__dict__.get('oldf')
    
    @oldf.setter
    def oldf(self, value):
        """设置旧文件路径"""
        self.__dict__['oldf'] = value

    @property
    def old(self):
        """获取旧文件路径"""
        return self.__dict__.get('old')
    
    @old.setter
    def old(self, value):
        """设置旧文件路径"""
        self.__dict__['old'] = value

    @property
    def new(self):
        """获取新文件路径"""
        return self.__dict__.get('new')
    
    @new.setter
    def new(self, value):
        """设置新文件路径"""
        self.__dict__['new'] = value
    
    @property
    def repo_id(self):
        """获取repo_id"""
        return self.__dict__.get('repo_id')
    
    @repo_id.setter
    def repo_id(self, value):
        """设置repo_id"""
        self.__dict__['repo_id'] = value
        
    # ========== 公共方法 ==========
    def show_attributes(self):
        """显示所有属性的当前值"""
        print(f"ID: {self.id}")
        print(f"Context: {self.context}")
        print(f"Code Diff: {self.codediff}")
        print(f"New Hunk: {self.newhunk}")
        print(f"Messages: {self.comment}")
        print(f"Old File (f): {self.oldf}")
        print(f"Old File: {self.old}")
        print(f"New File: {self.new}")
        print(f"Repo ID: {self.repo_id}")

    def validate_all(self):
        """验证所有必需属性是否已设置"""
        required_attrs = ['context', 'oldf', 'new']
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(f"Required attribute '{attr}' not initialized")
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        clean_key = key.lstrip('_')
        return clean_key in self.__dict__