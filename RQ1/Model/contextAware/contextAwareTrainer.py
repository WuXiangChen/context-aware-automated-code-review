
'''
  导包区
'''
import logging
from Model._1_BaseTrainer import RefinementTrainer
from transformers import get_scheduler
from torch.optim import AdamW

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO)
logger = logging.getLogger("Output/logs/"+__name__)


class contextAwareREF(RefinementTrainer):
  def __init__(self, args, data_file: str, model=None,  eval_=False, ref_node_list=None):
    logger.info(f"nodeclass in codereviewerREF, its length is {len(ref_node_list)}")
    super().__init__(args=args, data_file=data_file, model=model, eval_=eval_, ref_node_list=ref_node_list)
    del ref_node_list

  
  def run(self):
    return super().run()