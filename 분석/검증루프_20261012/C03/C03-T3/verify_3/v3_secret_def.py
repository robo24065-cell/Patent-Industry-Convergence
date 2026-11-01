import sys; sys.stdout.reconfigure(encoding='utf-8')
import pyarrow.parquet as pq
t=pq.read_table(sys.argv[1],columns=["year","secret_flag","perform_grp","perform_agent","gov_funds","total_funds"]).to_pandas()
y=t[t.year.astype(int)==2015]
print(y.perform_grp.astype(str).value_counts().head(12).to_dict())
a=y.perform_agent.astype(str).str.contains("보안")|y.perform_grp.astype(str).str.contains("보안")
print("alt",int(a.sum()),"mismatch",int((a!=y.secret_flag.astype(bool)).sum()))
print("all-years secret",int(t.secret_flag.sum()))
