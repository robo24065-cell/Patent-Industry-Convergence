# -*- coding: utf-8 -*-
"""s02와 같은 정의(복사): 정규화·유사도·체크섬."""
import re, difflib
import numpy as np

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?\s*LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return np.nan
    return difflib.SequenceMatcher(None, a, b).ratio()


def jur_ok(d):
    if not re.fullmatch(r"\d{13}", d or ""):
        return None
    s = sum(int(d[i]) * (1 if i % 2 == 0 else 2) for i in range(12))
    return (10 - s % 10) % 10 == int(d[12])


def biz_ok(d):
    if not re.fullmatch(r"\d{10}", d or ""):
        return None
    w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    s = sum(int(d[i]) * w[i] for i in range(9)) + (int(d[8]) * 5) // 10
    return (10 - s % 10) % 10 == int(d[9])


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def tvd(a, b):
    import pandas as pd
    pa = pd.Series(a).fillna("NA").value_counts(normalize=True)
    pb = pd.Series(b).fillna("NA").value_counts(normalize=True)
    ix = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(ix, fill_value=0) - pb.reindex(ix, fill_value=0)).abs().sum())
