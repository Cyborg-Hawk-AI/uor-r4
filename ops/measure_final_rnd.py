import subprocess, json, re, statistics
def energy_pkg(args):
    r = subprocess.run(['sudo','-S','perf','stat','-e','power/energy-pkg/','--']+args,
                       input='test\n', capture_output=True, text=True)
    m = re.search(r'([0-9.]+) Joules', r.stderr); s = re.search(r'([0-9.]+) seconds', r.stderr)
    return (float(m.group(1)), float(s.group(1)))
i_j, i_s = energy_pkg(['sleep','4']); idle_rate = i_j/i_s
N=2000; total_tokens = N * 19
res=[]
for rep in (1,2,3):
    wj, ws = energy_pkg(['bash','/home/yaz/loop.sh', str(N)])
    res.append((wj, ws))
gross_j = [wj/total_tokens for wj,ws in res]
net_j   = [(wj - idle_rate*ws)/total_tokens for wj,ws in res]
out={'label':'uor-r4-stack sustained generate loop, RAPL pkg, rnd (Linux x86_64)',
     'idle_rate_w': round(idle_rate,3),
     'work_rates_w': [round(wj/ws,3) for wj,ws in res],
     'net_rates_w': [round(wj/ws-idle_rate,3) for wj,ws in res],
     'tokens_measured': total_tokens,
     'gross_j_per_tok': {'mean': statistics.mean(gross_j),'min':min(gross_j),'max':max(gross_j)},
     'net_j_per_tok': {'mean': statistics.mean(net_j),'min':min(net_j),'max':max(net_j),
        'ci95': 1.96*statistics.stdev(net_j)/(len(net_j)**0.5) if len(net_j)>1 else 0},
     'note':'idle-rate subtraction (their energy_per_token.py shape); spawn overhead inside loop reported gross; small synthetic stack'}
print(json.dumps(out, indent=2))
