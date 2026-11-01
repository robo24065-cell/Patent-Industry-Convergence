import sys, glob, re, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
kr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("cache/etsi/kr_*.parquet"))], ignore_index=True)
def shape(s):
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", str(s)))
for c in ["Original_Publication_Number", "Normalized_Patent", "Patent_Type", "PBPA_APP_DATE"]:
    v2 = kr[c].drop_duplicates()
    print("==", c, len(v2))
    print(v2.map(shape).value_counts().head(12).to_string())
    print(v2.head(6).tolist())
v = kr.PUBL_NUMBER.drop_duplicates(); sh = v.map(shape)
for s in ["999999","9999999","9999-9999999","99-9999999","AA9999999 A9","999999999","AA 999999"]:
    print(s, v[sh==s].head(6).tolist())
k = pd.read_parquet(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/공모전데이터/작업데이터/kr_patents.parquet")
print(k.dtypes); print(k.head(2).T)
print(len(k), k["출원번호"].astype(str).duplicated().sum(), k["세세분류코드"].nunique())
print(k["출원연도"].describe())
print(k["등록번호"].astype(str).sample(8,random_state=0).tolist(), k["공개번호"].astype(str).sample(8,random_state=0).tolist())
