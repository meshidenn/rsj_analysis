from collections import defaultdict
from typing import Dict, Set

import pandas as pd
import numpy as np

MEDIAN = "median"
QUARTILE = "quartile"
Epsilon = 1e-03


class Binner:
    def binning(self, diff_rsj: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        bin_rsj = self._binning()
        df_rsj_info_bin = self.gather_df(diff_rsj, bin_rsj)
        return df_rsj_info_bin
    
    def gather_df(self, diff_rsj: Dict[str, Dict[str, float]], bin_rsj: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        rsj_info_bin = []
        for qid, diff_rsj_q in diff_rsj.items():
            for t in diff_rsj_q:
                rsj_info_bin.append((diff_rsj_q[t], bin_rsj[qid][t]))
        df_rsj_info_bin = pd.DataFrame(rsj_info_bin, columns=["ΔRSJ", "bin"])
        return df_rsj_info_bin
    
    def _binning(self):
        raise NotImplementedError


class RSJThresholdBinner(Binner):
    def __init__(self, rsj, rsj_threshold_type: str) -> None:
        self.rsj_threshold = self.rsj_threshold(rsj, rsj_threshold_type)
        self.rsj = rsj        
    
    def _binning(self) -> Dict[str, Dict[str, str]]:
        bin_rsj = {}
        for qid, rsj_uq in self.rsj.items():
            if qid not in bin_rsj:
                bin_rsj[qid] = defaultdict(float)
            for t, v in rsj_uq.items():
                if t not in rsj_uq:
                    bin_rsj[qid][t] = "OOV"
                    continue
                if rsj_uq[t] >= self.rsj_threshold:
                    bin_rsj[qid][t] = "HighRSJ"
                else:
                    bin_rsj[qid][t] = "LowRSJ"
        return bin_rsj    
    
    def rsj_threshold(self, rsj: Dict[str, Dict[str, float]], threshold_type: str) -> float:
        all_rsj_in_q = [v for qid, term_value in rsj.items() for t, v in term_value.items()],
        if threshold_type == MEDIAN:
            return np.median(all_rsj_in_q)
        elif threshold_type == QUARTILE:
            q75, q25 = np.percentile(all_rsj_in_q, [75, 25])
            return q75

        
class RSJIDFThresholdBinner(RSJThresholdBinner):
    def __init__(self, idf: Dict[str, float], rsj: Dict[str, Dict[str, float]], all_query_terms: Set[str], idf_threshold_type: str, rsj_threshold_type: str) -> float:
        self.rsj_threshold = self.rsj_threshold(rsj, rsj_threshold_type)
        self.idf_threshold = self.idf_threshold(idf, all_query_terms, idf_threshold_type)
        self.idf = idf
        self.rsj = rsj
    
    def _binning(self):
        bin_rsj = {}
        for t in self.idf:
            for qid, rsj_uq in self.rsj.items():
                if qid not in bin_rsj:
                    bin_rsj[qid] = defaultdict(float)
                if t not in rsj_uq:
                    bin_rsj[qid][t] = "OOV"
                    continue
                if self.idf[t] < self.idf_threshold:
                    if rsj_uq[t] > self.rsj_threshold:
                        bin_rsj[qid][t] = "HighRSJ:LowIDF"
                    else:
                        bin_rsj[qid][t] = "LowRSJ:LowIDF"
                else:
                    if rsj_uq[t] > self.rsj_threshold:
                        bin_rsj[qid][t] = "HighRSJ:HighIDF"
                    else:
                        bin_rsj[qid][t] = "LowRSJ:HighIDF"
        return bin_rsj
        
    def idf_threshold(self, idf: Dict[str, float], all_query_terms: Set[str], threshold_type: str) -> float:
        all_idf_in_q = [v for t, v in idf.items() if t in all_query_terms] 
        if threshold_type == MEDIAN:
            return np.median(all_idf_in_q)
        elif threshold_type == QUARTILE:
            q75, q25 = np.percentile(all_idf_in_q, [75, 25])
            return q75
        else:
            raise ValueError(f"{threshold_type} does not exist.")


class NumBinner(Binner):
    def __init__(self, rsj: Dict[str, Dict[str, float]], num: int = 4) -> None:
        self.rsj = rsj
        self.num = num
        
    def _binning(self) -> Dict[str, Dict[str, float]]:
        all_rsj_in_q = []
        all_qid = []
        all_t = []
        for qid, term_value in self.rsj.items():
            for t, v in term_value.items():
                all_qid.append(qid)
                all_t.append(t)
                all_rsj_in_q.append(v)
        
        bins = np.linspace(min(all_rsj_in_q), max(all_rsj_in_q), self.num)
        bins[-1] += Epsilon
        bined_rsj_in_q = np.digitize(all_rsj_in_q, bins)
        bin_names = {i+1: (round(bins[i], 2), round(bins[i+1], 2)) for i in range(len(bins)-1)}
        named_bined_rsj_in_q = [bin_names[i] for i in bined_rsj_in_q]
        bin_rsj = {}
        for qid, t, name_bin in zip(all_qid, all_t, named_bined_rsj_in_q):
            if qid not in bin_rsj:
                bin_rsj[qid] = defaultdict(float)
            bin_rsj[qid][t] = name_bin
        return bin_rsj
