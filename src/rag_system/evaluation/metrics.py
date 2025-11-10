"""Evaluation metrics for RAG system."""

import numpy as np
from typing import List, Dict, Any, Optional
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from ..base import BaseEvaluator, RetrievalResult, RAGResponse
from ...utils.logger import get_logger

logger = get_logger()


class RetrievalMetrics(BaseEvaluator):
    """Metrics for evaluating retrieval quality."""

    def evaluate(
        self,
        predictions: List[List[RetrievalResult]],
        references: List[List[str]],
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate retrieval predictions against ground truth.

        Args:
            predictions: List of retrieved results for each query
            references: List of relevant document IDs for each query
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        metrics = {
            "precision@k": self._precision_at_k(predictions, references),
            "recall@k": self._recall_at_k(predictions, references),
            "mrr": self._mean_reciprocal_rank(predictions, references),
            "ndcg": self._ndcg(predictions, references)
        }

        return metrics

    def _precision_at_k(
        self,
        predictions: List[List[RetrievalResult]],
        references: List[List[str]]
    ) -> float:
        """Calculate Precision@K.

        Args:
            predictions: Retrieved results
            references: Relevant document IDs

        Returns:
            Precision@K score
        """
        precisions = []

        for pred, ref in zip(predictions, references):
            retrieved_ids = [r.document.doc_id for r in pred]
            relevant_retrieved = len(set(retrieved_ids) & set(ref))
            precision = relevant_retrieved / len(pred) if pred else 0.0
            precisions.append(precision)

        return float(np.mean(precisions))

    def _recall_at_k(
        self,
        predictions: List[List[RetrievalResult]],
        references: List[List[str]]
    ) -> float:
        """Calculate Recall@K.

        Args:
            predictions: Retrieved results
            references: Relevant document IDs

        Returns:
            Recall@K score
        """
        recalls = []

        for pred, ref in zip(predictions, references):
            if not ref:  # No relevant documents
                continue

            retrieved_ids = [r.document.doc_id for r in pred]
            relevant_retrieved = len(set(retrieved_ids) & set(ref))
            recall = relevant_retrieved / len(ref)
            recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0

    def _mean_reciprocal_rank(
        self,
        predictions: List[List[RetrievalResult]],
        references: List[List[str]]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        Args:
            predictions: Retrieved results
            references: Relevant document IDs

        Returns:
            MRR score
        """
        reciprocal_ranks = []

        for pred, ref in zip(predictions, references):
            retrieved_ids = [r.document.doc_id for r in pred]

            # Find rank of first relevant document
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in ref:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))

    def _ndcg(
        self,
        predictions: List[List[RetrievalResult]],
        references: List[List[str]],
        k: Optional[int] = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG).

        Args:
            predictions: Retrieved results
            references: Relevant document IDs
            k: Cutoff for NDCG@K

        Returns:
            NDCG score
        """
        ndcg_scores = []

        for pred, ref in zip(predictions, references):
            retrieved_ids = [r.document.doc_id for r in pred]
            if k:
                retrieved_ids = retrieved_ids[:k]

            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in ref:
                    dcg += 1.0 / np.log2(i + 1)

            # Calculate IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(ref), len(retrieved_ids))):
                idcg += 1.0 / np.log2(i + 2)

            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores))


class GenerationMetrics(BaseEvaluator):
    """Metrics for evaluating generation quality."""

    def __init__(self):
        """Initialize generation metrics."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate generated answers against references.

        Args:
            predictions: Generated answers
            references: Reference answers
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")

        metrics = {}

        # ROUGE scores
        rouge_scores = self._rouge_scores(predictions, references)
        metrics.update(rouge_scores)

        # BERT Score
        try:
            bert_scores = self._bert_scores(predictions, references)
            metrics.update(bert_scores)
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")

        return metrics

    def _rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary of ROUGE scores
        """
        rouge1_f, rouge2_f, rougeL_f = [], [], []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougeL_f.append(scores['rougeL'].fmeasure)

        return {
            "rouge1": float(np.mean(rouge1_f)),
            "rouge2": float(np.mean(rouge2_f)),
            "rougeL": float(np.mean(rougeL_f))
        }

    def _bert_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate BERTScore.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary of BERTScore metrics
        """
        P, R, F1 = bert_score(
            predictions,
            references,
            lang="en",
            verbose=False
        )

        return {
            "bert_score_precision": float(P.mean()),
            "bert_score_recall": float(R.mean()),
            "bert_score_f1": float(F1.mean())
        }


class RAGEvaluator:
    """Complete evaluator for RAG system."""

    def __init__(self):
        """Initialize RAG evaluator."""
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()

    def evaluate_retrieval(
        self,
        retrieved_results: List[List[RetrievalResult]],
        relevant_docs: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.

        Args:
            retrieved_results: Retrieved documents for each query
            relevant_docs: Relevant document IDs for each query

        Returns:
            Retrieval metrics
        """
        logger.info("Evaluating retrieval performance...")
        metrics = self.retrieval_metrics.evaluate(retrieved_results, relevant_docs)
        logger.info(f"Retrieval metrics: {metrics}")
        return metrics

    def evaluate_generation(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Evaluate generation performance.

        Args:
            generated_answers: Generated answers
            reference_answers: Reference answers

        Returns:
            Generation metrics
        """
        logger.info("Evaluating generation performance...")
        metrics = self.generation_metrics.evaluate(generated_answers, reference_answers)
        logger.info(f"Generation metrics: {metrics}")
        return metrics

    def evaluate_end_to_end(
        self,
        rag_responses: List[RAGResponse],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate end-to-end RAG performance.

        Args:
            rag_responses: RAG system responses
            ground_truth: Ground truth data with queries, answers, and relevant docs

        Returns:
            Complete evaluation metrics
        """
        logger.info("Evaluating end-to-end RAG performance...")

        # Extract components
        generated_answers = [resp.answer for resp in rag_responses]
        reference_answers = [gt['answer'] for gt in ground_truth]

        retrieved_results = [resp.source_documents for resp in rag_responses]
        relevant_docs = [gt['relevant_docs'] for gt in ground_truth]

        # Evaluate
        retrieval_metrics = self.evaluate_retrieval(retrieved_results, relevant_docs)
        generation_metrics = self.evaluate_generation(generated_answers, reference_answers)

        # Combine metrics
        all_metrics = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics
        }

        logger.info("End-to-end evaluation complete")
        return all_metrics

    def evaluate_answer_relevancy(
        self,
        query: str,
        answer: str,
        context: str
    ) -> float:
        """Evaluate answer relevancy to query.

        Simple heuristic: check if answer contains query terms.

        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context

        Returns:
            Relevancy score (0-1)
        """
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        overlap = len(query_terms & answer_terms)
        score = overlap / len(query_terms) if query_terms else 0.0

        return min(score, 1.0)

    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """Evaluate if answer is faithful to context.

        Simple heuristic: check overlap between answer and context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Faithfulness score (0-1)
        """
        answer_terms = set(answer.lower().split())
        context_terms = set(context.lower().split())

        overlap = len(answer_terms & context_terms)
        score = overlap / len(answer_terms) if answer_terms else 0.0

        return min(score, 1.0)
