import pandas as pd
import pickle

attacklog = pd.read_csv('./attack.log')
victimlog = pd.read_csv('./pow.log')
avg = attacklog['accesstime'].mean()
af = attacklog[(attacklog['accesstime'] < 2*avg) & (attacklog['accesstime'] > avg//2)]
#startidx = victimlog.loc[victimlog['tag']=='pow start'].index[-1]
#victimlog = victimlog[startidx+1:]

traces = []
#avg = victimlog['iteration'].mean()
onetb = victimlog[victimlog['bit']==1]
meandur = onetb['iteration'].mean()
onetb2 = onetb[onetb['iteration']>meandur]
duration = onetb2['iteration'].mean()

for idx in victimlog.index:
    istart = victimlog['time'][idx]
    #duration = victimlog['iteration'][idx]
    trace = af.loc[(af['time'] > istart) & (af['time'] < istart + duration)]['accesstime']
    #print(len(trace))
    bit = victimlog['bit'][idx]
    traces.append((bit, trace))
print(attacklog['direction'][1])

datalist = []
try:
    with open(attacklog['direction'][1]+'.pkl','rb') as f:
        datalist = pickle.load(f)
except:
    pass
datalist.append(traces)
with open(attacklog['direction'][1]+'.pkl','wb') as f: 
    pickle.dump(datalist, f)