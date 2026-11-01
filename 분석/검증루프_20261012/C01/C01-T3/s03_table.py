import json,os
d=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C01\C01-T3"
pt=json.load(open(os.path.join(d,"s02_point.json"),encoding="utf-8"))
cis={s:json.load(open(os.path.join(d,f"s02_{s}.json"),encoding="utf-8")) for s in ["owner","bundle","bundle30"]}
for s in cis: print(s, cis[s]["B_done"], cis[s]["n_clusters"])
for lab,v in pt.items():
    print("==",lab, "n",v["n_eval"],"pos",v["pos"],"k",v["k1pct"], "hits1%", v["hits1pct_champ"],v["hits1pct_M0"],v["hits1pct_SB"], "hit100",v["hit100_champ"],v["hit100_M0"],v["hit100_SB"])
    for mt in ["auc_champ","auc_M0","auc_SB","skill_M0","skill_SB","cap_champ","cap_M0","cap_SB","dcap_M0","dcap_SB"]:
        print(f"  {mt:10s} {v[mt]:.3f}", "  ".join(f"{s}[{cis[s][lab][mt][0]:.3f},{cis[s][lab][mt][1]:.3f}]" for s in cis))
    print("   P(dcap_SB<=0)", {s:round(cis[s][lab]["share_dcap_SB_le0"],3) for s in cis}, "P(skillSB<=0)", {s:round(cis[s][lab]["share_skill_SB_le0"],3) for s in cis})
