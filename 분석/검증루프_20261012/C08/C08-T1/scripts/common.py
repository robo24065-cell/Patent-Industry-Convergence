# -*- coding: utf-8 -*-
import re, unicodedata, os
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(ROOT, r"분석\모델경쟁_20260927")
WD = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1")
CACHE = os.path.join(WD, "cache")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
MAP5 = os.path.join(MC, r"_shared_report\fusion_advantage\map5")
SPLIT = os.path.join(MC, r"_shared_report\fusion_advantage\split34")
CCS = ["KR", "US", "JP", "CN", "EP"]

_KO = ["주식회사", "(주)", "㈜", "유한책임회사", "유한회사", "재단법인", "사단법인", "가부시키가이샤", "가부시끼가이샤", "가부시키 가이샤", "가부시끼 가이샤",
       "인코포레이티드", "코포레이션", "컴퍼니 리미티드", "컴퍼니리미티드", "리미티드", "엘엘씨", "게엠베하", "아크티엔게젤샤프트"]
_JC = ["株式会社", "有限会社", "合同会社", "股份有限公司", "有限责任公司", "有限公司"]
_LAT = ["kabushiki kaisha", "aktiengesellschaft", "incorporated", "corporation", "company", "limited", "l.l.c.", "s.p.a.", "(publ)",
        "co.", "co", "ltd.", "ltd", "inc.", "inc", "corp.", "corp", "llc", "l.p.", "lp", "gmbh", "ag", "s.a.", "sa", "plc", "kk", "n.v.", "nv",
        "b.v.", "bv", "oy", "ab", "spa", "sas", "se"]
_LAT_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted(_LAT, key=len, reverse=True)) + r")(?![a-z0-9])")
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def norm_n2(s):
    if s is None:
        return None
    s0 = unicodedata.normalize("NFKC", str(s)).strip().lower()
    if not s0:
        return None
    s1 = s0
    for t in _KO + _JC:
        s1 = s1.replace(t, " ")
    s1 = _LAT_RE.sub(" ", s1)
    s1 = _PUNCT.sub("", s1)
    return s1 if s1 else _PUNCT.sub("", s0) or s0
