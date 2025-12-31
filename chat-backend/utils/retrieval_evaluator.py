from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from rag_pipeline import retrieve

class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics:
    - MRR (Mean Reciprocal Rank)
    - Recall@K
    - Precision@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    """
    
    @staticmethod
    def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant document
        """
        for idx, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / idx
        return 0.0
    
    @staticmethod
    def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@K: Proportion of relevant docs in top K
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        num_relevant_retrieved = len(retrieved_k & relevant_set)
        return num_relevant_retrieved / len(relevant_set)
    
    @staticmethod
    def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@K: Proportion of relevant docs among top K retrieved
        """
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        num_relevant_retrieved = len(retrieved_k & relevant_set)
        return num_relevant_retrieved / k
    
    @staticmethod
    def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Considers position of relevant documents
        """
        if not relevant_ids or k == 0:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        # DCG: sum of (relevance / log2(position + 1))
        dcg = 0.0
        for idx, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(idx + 1)
        
        # IDCG: ideal DCG (all relevant docs at top)
        idcg = 0.0
        for idx in range(1, min(len(relevant_ids), k) + 1):
            idcg += 1.0 / math.log2(idx + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @classmethod
    def evaluate_retrieval(
        cls,
        collection,
        test_queries: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval on a set of test queries
        
        Args:
            collection: ChromaDB collection
            test_queries: List of dicts with 'query' and 'relevant_doc_ids'
            top_k: Number of documents to retrieve
        
        Returns:
            Dict with aggregated metrics
        """
        mrr_scores = []
        recall_scores = []
        precision_scores = []
        ndcg_scores = []
        
        for test_query in test_queries:
            query = test_query['query']
            relevant_ids = test_query.get('relevant_doc_ids', [])
            
            # Retrieve documents
            results = retrieve(collection, query, top_k=top_k)
            retrieved_ids = [r['id'] for r in results]
            
            # Calculate metrics
            mrr = cls.calculate_mrr(retrieved_ids, relevant_ids)
            recall = cls.calculate_recall_at_k(retrieved_ids, relevant_ids, top_k)
            precision = cls.calculate_precision_at_k(retrieved_ids, relevant_ids, top_k)
            ndcg = cls.calculate_ndcg_at_k(retrieved_ids, relevant_ids, top_k)
            
            mrr_scores.append(mrr)
            recall_scores.append(recall)
            precision_scores.append(precision)
            ndcg_scores.append(ndcg)
        
        # Calculate averages
        n = len(test_queries)
        return {
            'num_queries': n,
            'top_k': top_k,
            'mrr': sum(mrr_scores) / n if n > 0 else 0.0,
            'recall_at_k': sum(recall_scores) / n if n > 0 else 0.0,
            'precision_at_k': sum(precision_scores) / n if n > 0 else 0.0,
            'ndcg_at_k': sum(ndcg_scores) / n if n > 0 else 0.0,
            'details': [
                {
                    'query': test_queries[i]['query'],
                    'mrr': mrr_scores[i],
                    'recall_at_k': recall_scores[i],
                    'precision_at_k': precision_scores[i],
                    'ndcg_at_k': ndcg_scores[i]
                }
                for i in range(n)
            ]
        }
