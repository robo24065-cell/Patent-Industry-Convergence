import pandas as pd, re
from r4_common import *
ow = pd.read_parquet(f"{CACHE}/reg_owner_lt2023.parquet")
PUB = re.compile(r"대학교|산학협력단|산학협력|과학기술원|대학원|연구원|연구소|연구재단|국립|진흥원|대학|Institute|University", re.I)
FIRM = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임|합자회사|CO\.|LTD|INC|CORP|GMBH|주\)", re.I)
ow["pub"] = ow.ownerName.fillna("").str.contains(PUB) & ~ow.ownerName.fillna("").str.contains(FIRM)
x = ow[ow.pub].ownerName.value_counts()
print(len(x)); print(x.head(60).to_string())
print(ow[ow.ownerName.fillna("").str.contains("연구소") & ow.pub].ownerName.value_counts().head(30).to_string())
print(ow.cs.value_counts().head(20))
