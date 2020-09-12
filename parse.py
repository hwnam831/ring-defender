import pandas as pd
import pickle

attacklog = pd.read_csv('./attack.log')
victimlog = pd.read_csv('./victim.log')

startidx = victimlog.loc[victimlog['tag']=='pow start'].index[-1]
victimlog = victimlog[startidx+1:]

traces = []
avg = victimlog['iteration'].mean()

for idx in victimlog.index:
    istart = victimlog['time'][idx]
    duration = victimlog['iteration'][idx]
    trace = attacklog.loc[(attacklog['time'] > istart) & (attacklog['time'] < istart + duration)]['accesstime']
    #print(len(trace))
    bit = int(duration > avg)
    traces.append((bit, trace))
print(attacklog['direction'][1])
with open(attacklog['direction'][1]+'.pkl','wb') as f:
    pickle.dump(traces, f)