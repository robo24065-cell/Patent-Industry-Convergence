import sys; sys.stdout.reconfigure(encoding='utf-8')
for f in sys.argv[1:]:
    t=open(f,encoding='utf-8').read(); print(f[-20:], len(t), repr(t[-120:]))
