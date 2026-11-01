# KSIC50 공개 정의문(항목명·색인어·예시)에서 품목 용어 사전을 만들고 과제 텍스트에 대응시키는 함수 (임무 B, 방법 b2)
import re, numpy as np, pandas as pd
STOP = {"제조", "제조업", "기타", "부분품", "부속품", "부품", "관련제품", "관련", "제품", "장치", "기기", "장비", "용품",
        "재료", "물질", "처리", "가공", "기계", "기구", "전용", "일반", "각종", "유사", "신품", "그외", "조립", "연관",
        "제외", "시험", "농업", "건설", "영양", "재생", "입상", "액상", "보도", "연성", "경보", "선반", "전류", "저항",
        "직류", "교류", "계기", "합금", "정련", "도금", "착색", "크림", "거품"}
LATIN_STOP = {"FOUNDRY", "BRUSH", "MONOLITHIC", "IC", "PC", "TV", "MOS", "DYNAMIC", "STATIC", "RANDOM", "ACCESS", "MEMORY"}
def _clean(t):
    t = re.sub(r"(제조업|제조|임가공|\(.*?\))", " ", t)
    t = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", t)
    return re.sub(r"\s+", "", t).upper()
def build_terms(ksic_defs):
    ko, la = {}, {}
    for d in ksic_defs:
        k = d["code"]
        name_terms = {_clean(s) for s in re.split(r",|및", d["codenmKr"])}
        src = re.split(r",|및", d["codenmKr"]) + re.split(r",|·", d.get("codeIndex") or "") + re.split(r"·|\t|,", d.get("codeExample") or "")
        for s in src:
            t = _clean(s)
            if t in STOP or len(t) < 2: continue
            if re.fullmatch(r"[A-Z0-9 ]+", t):
                if len(t) >= 3 and t not in LATIN_STOP: la.setdefault(t, set()).add(k)
                continue
            if len(t) == 2 and t not in name_terms: continue      # 2글자 한글 용어는 항목명에 있을 때만
            ko.setdefault(t, set()).add(k)
        for s in [d["codenmKr"], d.get("codeIndex") or "", d.get("codeExample") or ""]:
            for inner in re.findall(r"\(([^)]*)\)", s):
                for w in re.split(r",|/", inner):
                    w = w.strip().upper()
                    if re.fullmatch(r"[A-Z][A-Z0-9\-]{2,15}", w) and w not in LATIN_STOP:
                        la.setdefault(w, set()).add(k)
    rows = [(t, k, 1.0 / len(ks), "ko") for t, ks in ko.items() for k in ks] + \
           [(t, k, 1.0 / len(ks), "la") for t, ks in la.items() for k in ks]
    return pd.DataFrame(rows, columns=["term", "ksic", "w", "kind"])
def score(texts, term_df, KSIC):
    """과제 텍스트 → (n, 50) 점수. 한글 용어는 공백 제거 문자열에서 부분일치(긴 용어 우선), 영문 약어는 단어 경계 일치."""
    idx = {k: i for i, k in enumerate(KSIC)}
    W = {}
    for t, k, w in zip(term_df.term, term_df.ksic, term_df.w):
        W.setdefault(t, []).append((idx[k], w * min(len(t), 8) / 4.0))
    ko = sorted(term_df[term_df.kind == "ko"].term.unique(), key=len, reverse=True)
    la = sorted(term_df[term_df.kind == "la"].term.unique(), key=len, reverse=True)
    rko = re.compile("|".join(map(re.escape, ko)))
    rla = re.compile(r"(?<![A-Z0-9])(" + "|".join(map(re.escape, la)) + r")(?![A-Z0-9])")
    t0 = texts.fillna("").str.replace(r"<[^>]+>", " ", regex=True).str.upper()
    S = np.zeros((len(t0), len(KSIC)), dtype=np.float32)
    for i, s in enumerate(t0.values):
        hits = set(rko.findall(re.sub(r"[^0-9A-Z가-힣]", "", s))) | set(rla.findall(s))
        for h in hits:
            for j, w in W[h]:
                S[i, j] += w
    return S
