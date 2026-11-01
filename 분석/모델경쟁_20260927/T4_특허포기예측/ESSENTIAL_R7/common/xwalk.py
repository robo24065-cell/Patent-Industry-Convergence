# -*- coding: utf-8 -*-
r"""R7 공용: 제공 KSIC 라벨을 대체할 '기존 다른 데이터' 산업 배정 = 공식 IPC-KSIC 연계표(2013, 특허청).
- 연계표(읽기 전용): 공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx '전체연계표' (산업코드, IPC 서브그룹, 건수)
  산업코드는 KSIC 9차 기반 60개 집계 단위(예 C2610 반도체 제조업 = 3자리, C2041 = 4자리). 5자리 세세분류는 없다.
- 배정 규칙: 출원의 메인 IPC → (1) 서브그룹 정확 일치 → (2) 메인그룹(앞부분 'H01L 21') 합계 → (3) 서브클래스('H01L') 합계 중 건수 최대 산업코드.
- 메인 IPC 출처: 출원 서지(메인IPC). IPC는 KIPRIS 등 어느 특허 DB에나 있는 표준 서지항목이므로 '기존 데이터'로 본다(편의상 제공 레코드의 값을 읽음).
- 제공 50개 KSIC(10차 5자리) → 연계표 단위(9차) 대응표 K10_TO_XW: 10차 해설서 명칭으로 수작업 대응(아래 주석). 50개 → 25개 단위.
  즉 공식 연계표로는 같은 단위 안의 세세분류(예 26111 메모리 vs 26112 비메모리)를 원리적으로 구분할 수 없다.
python xwalk.py  → app_xw.parquet(KR 출원별 배정), xw_agreement.json(일치율)"""
import os, json, re
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
XLSX = os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx")
HERE = os.path.dirname(os.path.abspath(__file__))

# 10차 5자리 → 연계표 단위(9차). 근거: ksic50_definitions.json 명칭 vs 연계표 색인 명칭
K10_TO_XW = {
    "15220": "C1500",                      # 신발 부분품 → 가죽·가방·신발
    "20202": "C2030",                      # 합성수지 → (9차 203) 합성고무 및 플라스틱 물질
    "20411": "C2042",                      # 일반용 도료 → 잉크·페인트·코팅제
    "20423": "C2043",                      # 화장품 → 세제·화장품·광택제
    "20491": "C2049",                      # 감광재료 → 그 외 기타 화학제품
    "21220": "C2100",                      # 한의약품 → 의료용 물질 및 의약품
    "23992": "C2300",                      # 연마재 → 비금속 광물제품
    "24191": "C2410", "24219": "C2420",
    "25914": "C2590", "25922": "C2590", "25924": "C2590", "25932": "C2590",
    "26111": "C2610", "26112": "C2610", "26121": "C2610", "26129": "C2610",
    "26211": "C2620", "26212": "C2620", "26223": "C2620", "26224": "C2620", "26291": "C2620",
    "26322": "C2630", "26323": "C2630", "26422": "C2640", "26429": "C2640",
    "26519": "C2650", "26529": "C2650",
    "27199": "C2710", "27211": "C2720", "27212": "C2720", "27301": "C2730",
    "28111": "C2810", "28122": "C2810", "28511": "C2850", "28902": "C2890",
    "29131": "C2919", "29132": "C2919", "29133": "C2919", "29150": "C2919", "29162": "C2919",   # 일반 목적용 기계(사무용 이외)
    "29210": "C2920", "29229": "C2920", "29241": "C2920", "29280": "C2920",
    "30393": "C3000", "31202": "C3120", "31312": "C3130", "33301": "C3300", "33309": "C3300",
}


def norm_ipc(s):
    s = re.sub(r"\(.*?\)", "", str(s or "")).strip().upper()
    s = re.sub(r"\s+", " ", s)
    m = re.match(r"^([A-H]\d\d[A-Z])\s*(\d+)\s*/\s*(\d+)", s)
    if not m:
        m2 = re.match(r"^([A-H]\d\d[A-Z])", s)
        return (m2.group(1), None, None) if m2 else (None, None, None)
    return m.group(1), f"{m.group(1)} {int(m.group(2))}", f"{m.group(1)} {int(m.group(2))}/{m.group(3)}"


_T = {}


def tables():
    if _T:
        return _T
    x = pd.read_excel(XLSX, "전체연계표", header=0)
    x.columns = ["code", "ipc", "n"]
    x = x.dropna(subset=["code", "ipc"])
    x["n"] = pd.to_numeric(x.n, errors="coerce").fillna(0)
    t = x.ipc.map(norm_ipc)
    x["sc"] = [a[0] for a in t]; x["mg"] = [a[1] for a in t]; x["sg"] = [a[2] for a in t]
    for lvl in ["sg", "mg", "sc"]:
        g = x.dropna(subset=[lvl]).groupby([lvl, "code"]).n.sum().reset_index().sort_values([lvl, "n", "code"], ascending=[True, False, True])
        _T[lvl] = dict(zip(g.drop_duplicates(lvl)[lvl], g.drop_duplicates(lvl).code))
    _T["n_rows"] = len(x)
    return _T


def assign(ipcs):
    """ipcs: 메인 IPC 문자열 Series → (code, level)"""
    T = tables()
    codes, lv = [], []
    cache = {}
    for s in ipcs.fillna("").astype(str):
        if s in cache:
            c, l = cache[s]
        else:
            sc, mg, sg = norm_ipc(s)
            if sg and sg in T["sg"]: c, l = T["sg"][sg], "subgroup"
            elif mg and mg in T["mg"]: c, l = T["mg"][mg], "maingroup"
            elif sc and sc in T["sc"]: c, l = T["sc"][sc], "subclass"
            else: c, l = None, "none"
            cache[s] = (c, l)
        codes.append(c); lv.append(l)
    return pd.Series(codes, index=ipcs.index), pd.Series(lv, index=ipcs.index)


def load_app_xw():
    fn = os.path.join(HERE, "app_xw.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "출원연도", "메인IPC", "세세분류코드"])
    k.columns = ["appl_no", "ay", "main_ipc", "ksic"]
    k["appl_no"] = k.appl_no.astype(str); k["ksic"] = k.ksic.astype(str)
    a = k.drop_duplicates("appl_no")[["appl_no", "ay", "main_ipc"]].copy()
    a["xw"], a["xw_level"] = assign(a.main_ipc)
    # 제공 라벨(한 출원에 여러 KSIC 가능) → 연계표 단위 집합
    kk = k[["appl_no", "ksic"]].drop_duplicates()
    kk["parent"] = kk.ksic.map(K10_TO_XW)
    par = kk.groupby("appl_no").parent.apply(lambda s: "|".join(sorted(set(s.dropna()))))
    ks = kk.groupby("appl_no").ksic.apply(lambda s: "|".join(sorted(set(s))))
    a["prov_ksic"] = a.appl_no.map(ks); a["prov_parent"] = a.appl_no.map(par)
    a["agree"] = [bool(x) and bool(p) and x in p.split("|") for x, p in zip(a.xw, a.prov_parent)]
    a.to_parquet(fn, index=False)
    return a


def ksic_x_for(appl_nos):
    """출원번호 → 연계표 배정 코드(없으면 'XNA')"""
    a = load_app_xw().set_index("appl_no").xw
    return pd.Series(appl_nos).astype(str).map(a).fillna("XNA").values


if __name__ == "__main__":
    T = tables()
    a = load_app_xw()
    k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "세세분류코드"])
    k.columns = ["appl_no", "ksic"]; k["appl_no"] = k.appl_no.astype(str); k["ksic"] = k.ksic.astype(str)
    kk = k.drop_duplicates().merge(a[["appl_no", "xw"]], on="appl_no")
    kk["agree"] = kk.xw == kk.ksic.map(K10_TO_XW)
    per = kk.groupby("ksic").agg(n=("appl_no", "size"), agree=("agree", "mean"),
                                 top_xw=("xw", lambda s: s.value_counts().index[0] if s.notna().any() else None)).reset_index()
    per["parent"] = per.ksic.map(K10_TO_XW)
    per.to_csv(os.path.join(HERE, "xw_agreement_by_ksic.csv"), index=False, encoding="utf-8-sig")
    out = dict(
        crosswalk_rows=int(T["n_rows"]), n_xw_codes_used_by_50=len(set(K10_TO_XW.values())),
        n_apps=int(len(a)), level_share={k_: round(float(v), 4) for k_, v in a.xw_level.value_counts(normalize=True).items()},
        app_agree_rate=round(float(a.agree.mean()), 4),
        pair_agree_rate=round(float(kk.agree.mean()), 4),
        multi_label_app_share=round(float((a.prov_ksic.str.count(r"\|") > 0).mean()), 4),
        xw_outside_50_parents_share=round(float((~a.xw.isin(set(K10_TO_XW.values()))).mean()), 4),
        note="app_agree: 출원의 연계표 배정 코드가 그 출원의 제공 KSIC 라벨(들)의 상위 단위 중 하나와 같음. 5자리 수준 일치는 원리적으로 불가(50→25 단위).",
    )
    json.dump(out, open(os.path.join(HERE, "xw_agreement.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(per.sort_values("agree").to_string())
