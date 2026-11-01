# s2: 소멸등록 시차(개발 자료만)로 잠금 창 끝 E를 사전 규칙대로 정한다 + 시점 정합 점검(월별 '건수'만).
import os, json
import pandas as pd, numpy as np
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
df = pd.read_parquet(os.path.join(OUTD, "pat_base.parquet"))
res = {}
L = df[df.lapse_reg_d.notna()].copy()
L["lag"] = (L.lapse_reg_d - L.paid_thru).dt.days
dev = L[(L.paid_thru >= "2016-01-01") & (L.paid_thru <= "2022-12-31")]
q = dev.lag.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99]).round(0)
res["lag_dev_n"] = int(len(dev)); res["lag_dev_q"] = {str(k): float(v) for k, v in q.items()}
res["lag_dev_neg_share"] = float((dev.lag < 0).mean())
by_y = dev.groupby(dev.paid_thru.dt.year).lag.quantile(0.97)
res["lag_p97_by_paidyear"] = {int(k): float(v) for k, v in by_y.items()}
# 최근 연도(2023–2024 납부완료) 시차 — 관측 절단 때문에 짧아 보일 수 있음(참고)
rec = L[(L.paid_thru >= "2023-01-01") & (L.paid_thru <= "2024-12-31")]
res["lag_2023_24_p97_truncated"] = float(rec.lag.quantile(0.97))
P97 = float(dev.lag.quantile(0.97))
snap = pd.Timestamp("2026-08-01")
cands = pd.date_range("2025-01-31", "2025-12-31", freq="ME")
ok = [d for d in cands if (snap - d).days >= P97]
E = max(ok) if ok else None
res["P97_days"] = P97; res["E_rule"] = str(E.date()) if E is not None else None
res["Delta_days"] = int((snap - E).days) if E is not None else None
# 시점 정합: 소멸등록(불납) 월별 건수 2023-01 ~ 2026-08 (건수만)
m = L[L.lapse_reg_d >= "2023-01-01"].lapse_reg_d.dt.to_period("M").value_counts().sort_index()
res["lapse_reg_monthly"] = {str(k): int(v) for k, v in m.items()}
# 회복 비율: 납부완료 연도별 (소멸 후 회복 / 소멸)
L["restored"] = L.restore_reg_d.notna() & (L.restore_reg_d >= L.lapse_reg_d)
rr = L[L.paid_thru.dt.year.between(2012, 2024)].groupby(L.paid_thru.dt.year).restored.mean().round(4)
res["restore_rate_by_paidyear"] = {int(k): float(v) for k, v in rr.items()}
rlag = (L[L.restored].restore_reg_d - L[L.restored].lapse_reg_d).dt.days
res["restore_lag_q"] = {str(k): float(v) for k, v in rlag.quantile([0.5, 0.9, 0.97]).items()}
# 정합: 납부완료일이 2016–2022인데 소멸등록이 없고, 그 뒤 납부도 없는 특허(= 미기록 소멸 또는 자료 결손)
nolap = df[(df.paid_thru >= "2016-01-01") & (df.paid_thru <= "2022-12-31") & df.lapse_reg_d.isna() & df.expire_reg_d.isna()]
res["paidthru_16_22_no_lapse_no_expiry"] = int(len(nolap))
res["paidthru_16_22_total"] = int(((df.paid_thru >= "2016-01-01") & (df.paid_thru <= "2022-12-31")).sum())
res["nolap_expiry_before_paidthru"] = int((nolap.cndrtExptnDate_d <= nolap.paid_thru + pd.Timedelta(days=1)).sum())
# 소멸등록이 있는데 납부완료일이 소멸등록 뒤(모순)
res["lapse_before_paidthru"] = int((L.lag < 0).sum())
json.dump(res, open(os.path.join(OUTD, "s2_lag.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ["lag_dev_q", "P97_days", "E_rule", "Delta_days", "lag_dev_neg_share", "paidthru_16_22_no_lapse_no_expiry", "paidthru_16_22_total", "nolap_expiry_before_paidthru", "lapse_before_paidthru"]}, ensure_ascii=False))
