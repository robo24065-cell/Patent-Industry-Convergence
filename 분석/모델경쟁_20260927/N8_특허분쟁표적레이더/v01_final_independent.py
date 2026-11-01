# -*- coding: utf-8 -*-
"""최종시험 독립 재계산(final 실행 이후). n8lib의 extract/build_t0/final_labels/auc/topk를 쓰지 않고
등록원부(sqlite, mode=ro)에서 직접 (1) t0=2022 모집단 (2) 결과 레이블을 다시 만들고,
(3) final_preds_r6.parquet 점수로 AUC(Mann–Whitney 순위식)·skill·상위 k를 다시 계산해 final_result_r6.json과 대조한다.
(4) 군집 부트스트랩은 다른 시드(777)로 1,000회 다시 돌려 CI 안정성을 본다(n8lib.boot_ci 사용, 이 부분만 독립 아님).
출력: v01_final_independent.json"""
import sys, os, json, sqlite3
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import rankdata

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
E = "20221231"

# (1) 모집단: 등록 ≤ 2022-12-31, 소멸·말소 사건(≤ 2022-12-31) 없음, 존속만료 > 2022-12-31
pt = pd.read_sql("select rgst_no, rgstDate, cndrtExptnDate from patent", c)
ext = pd.read_sql(f"""select distinct rgst_no from right where rgstCsName in ('소멸등록','권리의 말소등록','말소등록')
                      and rgstCsDate is not null and rgstCsDate <> '' and rgstCsDate <= '{E}'""", c)
rg = pd.to_numeric(pt.rgstDate, errors="coerce"); ex = pd.to_numeric(pt.cndrtExptnDate, errors="coerce")
pop = pt[(rg <= int(E)) & (ex > int(E)) & ~pt.rgst_no.isin(set(ext.rgst_no))].rgst_no

# (2) 레이블: 이름 패턴으로 다시 찾는다(정확 일치 목록을 쓰지 않음)
ev = pd.read_sql("""select rgst_no, rgstCsDate d, rgstCsName cs from right
                    where rgstCsDate >= '20230101' and rgstCsDate <= '20241231'
                    and (rgstCsName like '%무효%예고%' or rgstCsName like '%권리범위%예고%')""", c)
names = ev.cs.value_counts().to_dict()
ev_ok = ev[ev.cs.isin(["무효(심판)예고등록", "권리범위(심판)예고등록"])]
ypos = set(ev_ok.rgst_no) & set(pop)

P = pd.read_parquet(os.path.join(HERE, "final_preds_r6.parquet"))
R = json.load(open(os.path.join(HERE, "final_result_r6.json"), encoding="utf-8"))
out = dict(pop_n_indep=int(len(pop)), pop_n_final=int(len(P)),
           pop_same_set=bool(set(pop) == set(P.rgst_no)),
           event_names_found=names, n_events_indep=int(len(ev_ok)), n_events_final=R["n_label_events"],
           n_event_patents=int(ev_ok.rgst_no.nunique()), n_event_patents_outside_pop=int(len(set(ev_ok.rgst_no) - set(pop))),
           pos_indep=int(len(ypos)), pos_final=int(P.y.sum()))
y_ind = P.rgst_no.isin(ypos).astype(int).values
out["labels_identical"] = bool((y_ind == P.y.values).all())

# 등록 ≤2022이지만 모집단 밖인 분쟁 특허는 왜 빠졌나
outside = sorted(set(ev_ok.rgst_no) - set(pop))
o = pt[pt.rgst_no.isin(outside)].copy()
o["reason"] = np.select([pd.to_numeric(o.rgstDate, errors="coerce") > int(E), o.rgst_no.isin(set(ext.rgst_no)),
                         pd.to_numeric(o.cndrtExptnDate, errors="coerce") <= int(E)], ["2023년 이후 등록", "원점 전 소멸·말소", "존속만료"], "기타")
out["outside_reasons"] = o.reason.value_counts().to_dict()
out["outside_not_in_patent_table"] = int(len(set(outside) - set(pt.rgst_no)))


def mw_auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def topk_hits(y, p, k):
    # 동점은 평균 처리: 경계 동점 묶음은 기대 적중으로 센다(정렬 순서 의존성 점검)
    thr = np.sort(p)[::-1][k - 1]
    above = p > thr; tie = p == thr
    return float(y[above].sum() + y[tie].sum() * (k - above.sum()) / tie.sum())


y = y_ind
rec = {}
for m in ["champ", "M0", "SB", "logit_champ", "logit_pilot"]:
    p = P[f"p_{m}"].values.astype(float)
    a = mw_auc(y, p)
    d = dict(auc=a, auc_final=R[f"{m}_auc"], auc_match=abs(a - R[f"{m}_auc"]) < 1e-9)
    for nm, fr in (("1pct", 0.01), ("5pct", 0.05), ("10pct", 0.10)):
        k = max(1, int(round(fr * len(y))))
        d[f"top{nm}_hits_tieavg"] = topk_hits(y, p, k); d[f"top{nm}_hits_final"] = R[f"{m}_top{nm}_hits"]
    d["hit100_tieavg"] = topk_hits(y, p, 100); d["hit100_final"] = R[f"{m}_hit100"]
    d["n_unique_scores"] = int(len(np.unique(p)))
    rec[m] = d
out["metrics"] = rec
out["skill_vs_M0"] = (rec["champ"]["auc"] - rec["M0"]["auc"]) / (1 - rec["M0"]["auc"])
out["skill_vs_SB"] = (rec["champ"]["auc"] - rec["SB"]["auc"]) / (1 - rec["SB"]["auc"])
out["skill_match"] = bool(abs(out["skill_vs_M0"] - R["skill_vs_M0"]) < 1e-9 and abs(out["skill_vs_SB"] - R["skill_vs_SB"]) < 1e-9)

# (4) 다른 시드 부트스트랩(군집 = 원점 당시 권리자). 군집키는 final 패널에 없어서 cache 대신 등록원부에서 다시 만든다.
ow = pd.read_sql(f"""select rgst_no, seq, ownerName nm, ownerRgstCsDate d from owner where ownerRgstCsDate is null or ownerRgstCsDate='' or ownerRgstCsDate <= '{E}'
                    union all select rgst_no, seq, utilityModelOwnerName nm, utilityModelOwnerRgstCsDate d from utilityModelOwner
                    where utilityModelOwnerRgstCsDate is null or utilityModelOwnerRgstCsDate='' or utilityModelOwnerRgstCsDate <= '{E}'""", c)
sys.path.insert(0, HERE)
from n8lib import norm_name, boot_ci
ow = ow.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no", keep="last")
cur = dict(zip(ow.rgst_no, ow.nm.map(norm_name)))
g = np.array([cur.get(r, "") or ("R" + r) for r in P.rgst_no])
out["n_clusters"] = int(len(np.unique(g)))
for ref, sd in (("M0", 777), ("SB", 778)):
    ci = boot_ci(y, P.p_champ.values, P[f"p_{ref}"].values, g, n=1000, seed=sd)
    out[f"skill_vs_{ref}_ci_seed{sd}"] = ci[0]; out[f"skill_vs_{ref}_ci_final"] = R[f"skill_vs_{ref}_ci"]
json.dump(out, open(os.path.join(HERE, "v01_final_independent.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
