from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import bert_score
import numpy as np

class MetricsEvaluator:
    def __init__(self):
        pass

    def calculate_corpus_bleu(self, predictions, references, n=4):
        tokenized_predictions = [item.split() for item in predictions]
        tokenized_references = [item.split() for item in references]
        return corpus_bleu(
            list_of_references=tokenized_predictions,
            hypotheses=tokenized_references,
            weights=(1 / n,) * n,
        )

    def calculate_bertscore(self, predictions, references):
        bs_p, bs_r, bs_f1 = bert_score.score(
            predictions,
            references,
            model_type="bert-base-multilingual-cased",
        )
        mean_bs_p = bs_p.mean().item()
        mean_bs_r = bs_r.mean().item()
        mean_bs_f1 = bs_f1.mean().item()
        return {
            'precision': mean_bs_p,
            'recall': mean_bs_r,
            'f1': mean_bs_f1
        }

    def calculate_meteor_score(self, predictions, references):
        tokenized_predictions = [item.split() for item in predictions]
        tokenized_references = [item.split() for item in references]
        score = [
            single_meteor_score(pred, ref)
            for pred, ref in zip(tokenized_predictions, tokenized_references)
        ]
        mean_score = np.mean(score)
        return mean_score

    def calculate_rouge_score(self, predictions, references):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        results = [
            scorer.score(ref, pred)['rougeL']
            for pred, ref in zip(predictions, references)
        ]
        mean_p = np.mean([res.precision for res in results])
        mean_r = np.mean([res.recall for res in results])
        mean_f1 = np.mean([res.fmeasure for res in results])
        return {
            'precision': mean_p,
            'recall': mean_r,
            'f1': mean_f1
        }

    def calculate_scores(self, predictions, references):
        bleu_score = self.calculate_corpus_bleu(predictions, references)
        rouge_score = self.calculate_rouge_score(predictions, references)
        meteor_score = self.calculate_meteor_score(predictions, references)
        bert_score = self.calculate_bertscore(predictions, references)
        return {
            'bleu': bleu_score,
            'rouge': rouge_score,
            'meteor': meteor_score,
            'bertscore': bert_score
        }