import os

glist=[]
plist=[]
with open("lts.s2s.train",encoding='utf8') as f:
    for line in f:
        t1,t2=line.strip().split('\t')
        glist.append(t1)
        plist.append(' '.join(t2.split(' ')[::-1]))

with open("fair-data2/train.g",'w',encoding='utf8') as f:
    f.write('\n'.join(glist))
with open("fair-data2/train.p",'w',encoding='utf8') as f:
    f.write('\n'.join(plist))


