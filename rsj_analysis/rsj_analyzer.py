from collections import defaultdict
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from .binner import Binner


class InvertedIndex:
    def __init__(self, corpus: Dict[str, Dict[str, str]], tokenizer: Any) -> None:
        self._n_doc = len(corpus)
        self.tokenizer = tokenizer
        self.index, self.df, self.idf = self._indexer(corpus, tokenizer)
        
    def _indexer(self, corpus: Dict[str, Dict[str, str]], tokenizer: Any) -> None:
        index_corpus = defaultdict(list)
        idf = defaultdict(float)
        for cid, doc in tqdm(corpus.items(), total=self._n_doc):
            text = doc["title"] + " " + doc["text"]
            t_doc = tokenizer(text)
            for t in set(t_doc):
                index_corpus[t].append(cid)
                
        df = {k: len(v) for k, v in index_corpus.items()}
                
        for v, freq in df.items():
            idf[v] = np.log(self.n_doc / freq)
        
        return index_corpus, df, idf
    
    @property
    def n_doc(self):
        return self._n_doc    

class RSJCalculator:
    def __init__(self, index: InvertedIndex, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> None:
        self.index = index
        self.t_queries, self.all_query_terms = self.tokenize_queries(queries, qrels, index.tokenizer)
        rel_qrels = self._extract_rel(qrels)
        self.rsj = self._calc_rsj(self.t_queries, rel_qrels)
        
    def tokenize_queries(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], tokenizer: Any) -> Tuple[Dict[str, List[str]], Set[str]]:
        t_queries = {}
        all_query_terms = set()
        for qid, _ in qrels.items():
            try:
                query = queries[qid]
            except KeyError:
                continue
            t_query = tokenizer(query)
            t_queries[qid] = t_query
            all_query_terms |= set(t_query)
        return t_queries, all_query_terms
    
    def _extract_rel(self, qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        rel_qrels = {}
        for qid, q_qrels in qrels.items():
            rel_q_qrels = {k: v for k, v in q_qrels.items() if v > 0}
            rel_qrels[qid] = rel_q_qrels
        return rel_qrels
            
    def _calc_rsj(self, t_queries: Dict[str, List[str]], relqrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        rsj = defaultdict(float)
        for qid, q_qrels in relqrels.items():
            t_query = t_queries[qid]
            rel_q_qrels = {k: v for k, v in q_qrels.items() if v > 0}
            Nr = len([i for i in q_qrels.values() if i > 0])
            if Nr < 1:
                continue
            wtq = defaultdict(float)
            for t in t_query:
                Dt = self.index.index[t]
                Nt = len(Dt)
                Ntr = 0
                for did in Dt:
                    if did in rel_q_qrels:
                        Ntr+= 1
                    
                N = self.index.n_doc
                wtq[t] = np.log(((Ntr + 0.5)*(N - Nt - Nr + Ntr + 0.5))/((Nr - Ntr + 0.5)*(Nt - Ntr + 0.5)) )
            rsj[qid] = wtq
        return rsj
        
    def calc_diff_rsj(self, search_result: Dict[str, float], top_K: int =100) -> Dict[str, Dict[str, float]]:
        topK_rels = {}
        for qid, rels in search_result.items():
            topK_rels[qid] = dict(sorted(rels.items(), key=lambda x: -x[1])[:top_K])
            
        rsj_s = self._calc_rsj(self.t_queries, topK_rels)
        diff_rsj = defaultdict(dict)
        for qid in self.rsj:
            if qid not in rsj_s:
                continue
            for t in self.t_queries[qid]:
                diff_rsj[qid][t] = rsj_s[qid][t] - self.rsj[qid][t]
                
        return diff_rsj
    
def analyze(search_result: Dict[str, Dict[str, float]], rsj_calculator: RSJCalculator, binner: Binner, top_K: int = 100) -> pd.DataFrame:
    diff_rsj = rsj_calculator.calc_diff_rsj(search_result, top_K=top_K)
    binned_rsj = binner.binning(diff_rsj)
    return binned_rsj