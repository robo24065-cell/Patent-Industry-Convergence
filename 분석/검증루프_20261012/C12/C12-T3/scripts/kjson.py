# -*- coding: utf-8 -*-
import json, re
def loadj(fp):
    txt = open(fp, 'rb').read().decode('utf-8', errors='replace')
    try:
        return json.loads(txt)
    except Exception:
        return json.loads(re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', txt))
