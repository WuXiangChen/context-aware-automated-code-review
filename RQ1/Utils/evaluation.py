from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class MetricsEvaluator:
  def __init__(self):
    self.smooth_fn = SmoothingFunction().method1
  
  def calculate_bleu(self, reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis.
    :param reference: List of reference sentences (list of tokens).
    :param hypothesis: Hypothesis sentence (list of tokens).
    :return: BLEU score.
    """
    return sentence_bleu([reference], hypothesis, smoothing_function=self.smooth_fn)

  def calculate_em(self, reference, hypothesis):
    """
    Calculate Exact Match (EM) score between reference and hypothesis.
    :param reference: Reference sentence (string).
    :param hypothesis: Hypothesis sentence (string).
    :return: 1 if exact match, otherwise 0.
    """
    return int(reference.strip() == hypothesis.strip())

# Example usage:
# if __name__ == "__main__":
#   evaluator = MetricsEvaluator()
  
#   # Example data
#   reference = "The cat is on the mat".split()
#   hypothesis = "The cat is on the mat".split()
  
#   bleu_score = evaluator.calculate_bleu(reference, hypothesis)
#   em_score = evaluator.calculate_em("The cat is on the mat", "The cat is on the mat")
  
#   print(f"BLEU Score: {bleu_score}")
#   print(f"Exact Match (EM) Score: {em_score}")