with open("lts.l.vocab") as f:
    llist=f.read().strip().split()
with open("fair-data/lts.l.vocab",'w') as f:
    f.write('\n'.join([t+' '+str(i) for i,t in enumerate(llist)]))
with open("lts.s.vocab") as f:
    llist=f.read().strip().split()
with open("fair-data/lts.s.vocab",'w') as f:
    f.write('\n'.join([t+' '+str(i) for i,t in enumerate(llist)]))
