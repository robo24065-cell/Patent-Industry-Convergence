# -*- coding: utf-8 -*-
"""사후 점검(채택 판단에 쓰지 않음): 강한 기준선·분해·부분집단·결과창 민감도·시드. 개발 원점만."""
import sys, re, warnings, numpy as np, pandas as pd
from r4_common import *
from r4_eval import SETS
warnings.filterwarnings("ignore")


def score(D, feats, ycol, dev, min_train, model="xgb", seed=0, mask=None):
    res = {}
    for v in dev:
        tr = D[(D.t0 <= v - 2) & (D.t0 >= min_train)]; te = D[D.t0 == v]
        p = fit_pred(tr[feats].astype(float), tr[ycol], te[feats].astype(float), model=model, seed=seed)
        if mask is not None:
            m = mask(te).values; res[v] = auc(te[ycol].values[m], p[m])
        else:
            res[v] = auc(te[ycol], p)
    return res


def cmp(name, a, b, dev):
    sk = [skill(a[v], b[v]) for v in dev]
    return dict(check=name, **{f"auc_{v}": round(a[v], 4) for v in dev}, **{f"ref_{v}": round(b[v], 4) for v in dev},
                mean_skill=round(float(np.mean(sk)), 4), wins=f"{sum(s > 0 for s in sk)}/{len(dev)}")


def i7():
    S = SETS["I7"]; D = pd.read_parquet(f"{CACHE}/I7_panel.parquet"); dev = S["dev"]; mt = S["min_train"]
    base = S["base"]; m1 = S["steps"][0][1]; m2 = S["steps"][1][1]; m3 = S["steps"][2][1]
    champ = base + m1 + m2 + m3
    out = []
    r_champ = score(D, champ, "y", dev, mt); r_m0 = score(D, base, "y", dev, mt)
    strong = base + ["log_app", "log_live", "n_ksic"]
    r_strong = score(D, strong, "y", dev, mt)
    out.append(cmp("챔피언 vs 강한 기준선(M0+특허 규모 3개)", r_champ, r_strong, dev))
    out.append(cmp("강한 기준선 vs M0", r_strong, r_m0, dev))
    r_noM2 = score(D, base + m1 + m3, "y", dev, mt)
    out.append(cmp("챔피언 vs NTIS 참여(M2) 제거", r_champ, r_noM2, dev))
    r_noM3 = score(D, base + m1 + m2, "y", dev, mt)
    out.append(cmp("챔피언 vs KSIC 연구비(M3) 제거", r_champ, r_noM3, dev))
    D["y1"] = (D.first_ntis == D.t0 + 1).astype(int)
    out.append(cmp("결과창 1년(t0+1만): 챔피언 vs M0", score(D, champ, "y1", dev, mt), score(D, base, "y1", dev, mt), dev))
    nps0 = lambda te: te.nps_present == 0
    out.append(cmp("NPS 미가입 기업만: 챔피언 vs M0", score(D, champ, "y", dev, mt, mask=nps0), score(D, base, "y", dev, mt, mask=nps0), dev))
    out.append(cmp("시드 1: 챔피언 vs M0", score(D, champ, "y", dev, mt, seed=1), score(D, base, "y", dev, mt, seed=1), dev))
    # 기술통계: 참여 신호 보유 여부별 결과율
    desc = D[D.t0.isin(dev)].groupby(["t0", "g_part_any"]).y.agg(["size", "mean"]).reset_index()
    return pd.DataFrame(out), desc


def i8():
    S = SETS["I8"]; D = pd.read_parquet(f"{CACHE}/I8_panel.parquet"); dev = S["dev"]; mt = S["min_train"]
    base = S["base"]; m1 = S["steps"][0][1]; m2 = S["steps"][1][1]
    champ = base + m1 + m2
    k50 = ["k50_assigned", "k50_score", "b1_score"]; rest = [x for x in m1 if x not in k50]
    r_champ = score(D, champ, "y", dev, mt); r_m0 = score(D, base, "y", dev, mt)
    out = []
    out.append(cmp("M0+KSIC50 텍스트 대응 3개만 vs M0", score(D, base + k50, "y", dev, mt), r_m0, dev))
    out.append(cmp("M0+단계·6T·분류(텍스트 대응 제외) vs M0", score(D, base + rest, "y", dev, mt), r_m0, dev))
    r_np = score(D, base + m2, "y", dev, mt)
    out.append(cmp("챔피언 vs M0+NPS(과제 내용 없이)", r_champ, r_np, dev))
    # 출원인코드 연결 가능 기업만(연결 불가 기업은 구조적으로 y=0) — 연결 여부는 스냅샷이라 피처로는 쓰지 않음
    ns = pd.read_parquet(f"{EXT}/r3nts_applicant_status.parquet", columns=["bizno", "aname"])
    SUF = re.compile(r"주식회사|유한회사|\(주\)|㈜|\(유\)", re.I)
    nrm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
    import duckdb
    nt = duckdb.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, any_value(research_agency) nm from read_parquet('{EXT.replace(chr(92), '/')}/ntis_project_raw.parquet')
      where year between 2010 and 2019 group by 1""").df()
    uq = ns.assign(n=ns.aname.map(nrm)).groupby("n").size(); uq = set(uq[uq == 1].index)
    linked = set(ns.bizno) | set(nt[nt.nm.map(nrm).isin(uq)].bz)
    lk = lambda te: te.bz.isin(linked)
    out.append(cmp("출원인코드 연결 기업만: 챔피언 vs M0", score(D, champ, "y", dev, mt, mask=lk), score(D, base, "y", dev, mt, mask=lk), dev))
    out.append(cmp("시드 1: 챔피언 vs M0", score(D, champ, "y", dev, mt, seed=1), score(D, base, "y", dev, mt, seed=1), dev))
    desc = D[D.t0.isin(dev)].assign(k50=lambda d: d.k50_assigned > 0).groupby(["t0", "k50"]).y.agg(["size", "mean"]).reset_index()
    desc["linked_share"] = np.nan
    lr = D[D.t0.isin(dev)].bz.isin(linked).groupby(D.t0).mean()
    return pd.DataFrame(out), desc, lr


def i6():
    D = pd.read_parquet(f"{CACHE}/I6b_panel.parquet"); D = D[D.y >= 0]
    d = D[D.t0.between(2014, 2020)]
    t = d.groupby(["t0", "gov"], dropna=False).y.agg(["size", "mean"]).reset_index()
    t2 = d[d.gov == 1].groupby(["t0", "g_firm"]).y.agg(["size", "mean"]).reset_index()
    return t, t2


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    which = sys.argv[1]
    if which == "I7":
        a, b = i7(); print(a.to_string(index=False)); print(b.round(4).to_string(index=False))
        a.to_csv(f"{OUT}/I7_robust.csv", index=False, encoding="utf-8-sig"); b.to_csv(f"{OUT}/I7_desc.csv", index=False, encoding="utf-8-sig")
    elif which == "I8":
        a, b, lr = i8(); print(a.to_string(index=False)); print(b.round(4).to_string(index=False)); print("linked share", lr.round(3).to_dict())
        a.to_csv(f"{OUT}/I8_robust.csv", index=False, encoding="utf-8-sig"); b.to_csv(f"{OUT}/I8_desc.csv", index=False, encoding="utf-8-sig")
    else:
        a, b = i6(); print(a.round(4).to_string(index=False)); print(b.round(4).to_string(index=False))
        a.to_csv(f"{OUT}/I6_desc_gov.csv", index=False, encoding="utf-8-sig"); b.to_csv(f"{OUT}/I6_desc_gfirm.csv", index=False, encoding="utf-8-sig")
