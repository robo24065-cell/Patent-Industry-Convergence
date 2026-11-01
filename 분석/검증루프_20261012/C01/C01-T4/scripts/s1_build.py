# s1: 특허 단위 기초표(읽기 전용 sqlite → 이번 바퀴 cache parquet). 결과 레이블은 만들지 않는다.
import sqlite3, os, time
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
t = time.time()
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
pat = pd.read_sql("select rgst_no, ksic, applDate, rgstDate, cndrtExptnDate, applTpcd, lastDspst from patent", con)
pay = pd.read_sql("select rgst_no, max(cast(lastAnnl as integer)) max_annl, count(*) n_pay, max(payDate) last_paydate from pay group by rgst_no", con)
rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName like '%소멸%' or rgstCsName like '%회복%'", con)
own = pd.read_sql("select rgst_no, min(ownerRgstCsDate) first_trf, count(*) n_trf from owner where ownerRgstCsName is not null and ownerRgstCsDate is not null group by rgst_no", con)
app = pd.read_sql("select rgst_no, applicantCd from applicant", con)
dup_right = pd.read_sql("select count(*) from (select rgst_no, rgstCsDate, rgstCsName, count(*) c from right group by 1,2,3 having c>1)", con).iloc[0, 0]
con.close()
lap = rt[(rt.rgstCsName == "소멸등록") & (rt.rgstCsReason.isin(["등록료불납", "연차료불납"]))].groupby("rgst_no").rgstCsDate.max().rename("lapse_reg")
exp_ = rt[(rt.rgstCsName == "소멸등록") & (rt.rgstCsReason.isin(["존속기간만료", "기간만료"]))].groupby("rgst_no").rgstCsDate.max().rename("expire_reg")
rec = rt[rt.rgstCsName.str.startswith("회복")].groupby("rgst_no").rgstCsDate.max().rename("restore_reg")
nlap = rt[(rt.rgstCsName == "소멸등록")].groupby("rgst_no").size().rename("n_lapse_ev")
df = pat.merge(pay, on="rgst_no", how="left").merge(lap, on="rgst_no", how="left").merge(exp_, on="rgst_no", how="left") \
        .merge(rec, on="rgst_no", how="left").merge(nlap, on="rgst_no", how="left").merge(own, on="rgst_no", how="left")
rg = pd.to_datetime(df.rgstDate, format="%Y%m%d", errors="coerce")
# 납부완료일 = 등록일 + max_annl 년 (같은 월일; 2/29는 2/28로)
def add_years(d, n):
    out = []
    for x, k in zip(d, n):
        if pd.isna(x) or pd.isna(k):
            out.append(pd.NaT); continue
        try:
            out.append(x.replace(year=x.year + int(k)))
        except ValueError:
            out.append(x.replace(year=x.year + int(k), day=28))
    return out
df["paid_thru"] = add_years(rg, df.max_annl)
df["rg"] = rg
for c in ["lapse_reg", "expire_reg", "restore_reg", "first_trf", "cndrtExptnDate", "applDate"]:
    df[c + "_d"] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
# 회복 뒤 소멸이 다시 없으면 소멸 무효
df["lapsed"] = df.lapse_reg_d.notna() & ~(df.restore_reg_d.notna() & (df.restore_reg_d >= df.lapse_reg_d))
df.drop(columns=[c for c in ["lapse_reg", "expire_reg", "restore_reg", "first_trf"]], inplace=True)
df.to_parquet(os.path.join(OUTD, "pat_base.parquet"), index=False)
app.to_parquet(os.path.join(OUTD, "pat_app.parquet"), index=False)
with open(os.path.join(OUTD, "s1_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"patents {len(df)}, pay missing {df.max_annl.isna().sum()}, paid_thru NaT {df.paid_thru.isna().sum()}\n")
    f.write(f"lapse(불납) {df.lapse_reg_d.notna().sum()}, lapsed after restore rule {df.lapsed.sum()}, restore {df.restore_reg_d.notna().sum()}, expire {df.expire_reg_d.notna().sum()}\n")
    f.write(f"multi 소멸 events per patent>1: {(df.n_lapse_ev>1).sum()}\n")
    f.write(f"right duplicate (rgst_no,date,name) groups: {dup_right}\n")
    f.write(f"transfer ever {df.first_trf_d.notna().sum()}\n")
    f.write(f"applicant rows {len(app)}, dup (rgst_no,applicantCd) {app.duplicated(['rgst_no','applicantCd']).sum()}\n")
    f.write(f"elapsed {time.time()-t:.1f}s\n")
