# -*- coding: utf-8 -*-
# 8단계: 이름 표기 이동(합병·공동출원 선택)이 3.1.3 C2620 'S1(1위 출원인 제외)'에 주는 영향
# C08-T1 정의 재현: 전 유형 고유 출원, 창 2008–09 대 2016–17, 두 창 합산 1위 출원인 제거, gw=(ln(b/2+1)-ln(a/2+1))/8
import sys, os, time, json
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np, pandas as pd
t0 = time.time(); res = {}
kr = pd.read_parquet(os.path.join(CACHE, "kr_meta.parquet"), columns=["appl_no", "appl_year", "ksic", "applicant"])
d = kr[kr.ksic.isin(["26211", "26212"]) & kr.appl_year.isin([2008, 2009, 2016, 2017])].copy()
d["w"] = np.where(d.appl_year <= 2009, "a", "b")
top = d.groupby(["ksic", "w", "applicant"]).size().rename("n").reset_index()
res["top5_by_ksic_window"] = {f"{k}_{w}": g.sort_values("n", ascending=False).head(5)[["applicant", "n"]].values.tolist() for (k, w), g in top.groupby(["ksic", "w"])}
# 계열·승계 병합 맵(공개 사실: 삼성모바일디스플레이·에스엘시디 → 삼성디스플레이 2012 합병; 삼성전자 LCD사업부 → 삼성디스플레이 2012 분할)
MERGE_SUCC = {"삼성모바일디스플레이주식회사": "삼성디스플레이 주식회사"}
MERGE_GROUP = {"삼성모바일디스플레이주식회사": "삼성계", "삼성디스플레이 주식회사": "삼성계", "삼성전자주식회사": "삼성계",
               "엘지디스플레이 주식회사": "LG계", "엘지전자 주식회사": "LG계"}


def gw_s1(df, key):
    out = {}
    for k in ["26211", "26212"]:
        x = df[df.ksic == k]
        tot = x.groupby(key).size().sort_values(ascending=False)
        t1 = tot.index[0]
        a0 = (x.w == "a").sum(); b0 = (x.w == "b").sum()
        y = x[x[key] != t1]
        a = (y.w == "a").sum(); b = (y.w == "b").sum()
        out[k] = dict(top1=t1, top1_n=int(tot.iloc[0]), gw_S0=round(100 * (np.log(b0 / 2 + 1) - np.log(a0 / 2 + 1)) / 8, 1),
                      gw_S1=round(100 * (np.log(b / 2 + 1) - np.log(a / 2 + 1)) / 8, 1))
    return out


d["k_raw"] = d.applicant.fillna("")
d["k_succ"] = d.k_raw.replace(MERGE_SUCC)
d["k_group"] = d.k_raw.map(MERGE_GROUP).fillna(d.k_raw)
res["S1_raw_name"] = gw_s1(d, "k_raw")
res["S1_successor_merged"] = gw_s1(d, "k_succ")
res["S1_group_merged"] = gw_s1(d, "k_group")
# 2008–09 OLED에서 삼성모바일디스플레이 몫
o = d[(d.ksic == "26212") & (d.w == "a")]
res["smd_share_oled_0809"] = round(float((o.applicant == "삼성모바일디스플레이주식회사").mean()), 3)
res["sdc_share_oled_1617"] = round(float((d[(d.ksic == "26212") & (d.w == "b")].applicant == "삼성디스플레이 주식회사").mean()), 3)
json.dump(res, open(os.path.join(CACHE, "s8_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
