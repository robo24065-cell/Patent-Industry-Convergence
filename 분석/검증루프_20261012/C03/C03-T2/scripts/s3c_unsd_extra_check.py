import pandas as pd, sys
c12 = pd.read_csv(sys.argv[1]+r"\cache\unsd_corr_17_12.csv", dtype=str); c07 = pd.read_csv(sys.argv[1]+r"\cache\unsd_corr_17_07.csv", dtype=str)
print("17-12 new not in 17-07 new:", sorted(set(c12.new)-set(c07.new)), " 17-07 new not in 17-12:", sorted(set(c07.new)-set(c12.new)))
print(c12[c12.new.isin(set(c12.new)-set(c07.new))])
import collections
print("old codes count", c12.old.nunique(), c07.old.nunique())
odd = [o for o in c12.old.unique() if not o[:2].isdigit() or int(o[:2])>97]
print("odd old", odd[:10])
print(c12[c12.old.str.startswith("9999") | c12.new.str.startswith("9999")])
