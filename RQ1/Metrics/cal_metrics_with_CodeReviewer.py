# 本节的主要目的，是对code comments generation 和 code refinement generation任务进行评估BLEU/CodeBLUE，和EM两个指标

from typing import List
from .smooth_bleu import bleu_fromstr

class EvaluationMetrics():
  # 对pred路径与gold路径的初始化
  def __init__(self, pred_path:str, gold_path:str):
    self.pred_path = pred_path
    self.gold_path = gold_path
    self.predictions = None
    self.golds = None

  def bleu_metrics_(self, rmstop=True):
    self.generatePreAndGold()
    return bleu_fromstr(self.predictions, self.golds, rmstop)  

  def em_metrics_(self):
    self.generatePreAndGold()
    em = 0
    for pred, gold in zip(self.predictions, self.golds):
        if " ".join(pred.split()) == " ".join(gold.split()):
            em += 1
    em = em / len(self.golds)
    return em
  
  def generatePreAndGold(self):
    if self.predictions==None or self.golds==None:
      self.read_predictions_and_golds()
  
  # 我需要一个方法来进行读取对应的文件内容，并将其进行相应的计算
  def read_predictions_and_golds(self):
    """
    Helper function to read predictions and golds from their respective files.
    Reads the content of the files and returns them as List[str].
    """
    with open(self.pred_path, 'r', encoding='utf-8') as pred_file:
        self.predictions = [line.strip() for line in pred_file.readlines()]
    
    with open(self.gold_path, 'r', encoding='utf-8') as gold_file:
        self.golds = [line.strip() for line in gold_file.readlines()]
