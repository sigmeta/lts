import json

prob_path="../fairseq3/teacher-test/"
file_path="lts_data/lts.s2s.test"
tgt_path="lts_data_prob/lts.s2s.test"
with open(file_path) as f:
    txt_list=f.read().strip().split('\n')

for i in range(len(txt_list)):
    if i%1000==0:
        print(i)
    with open(prob_path+str(i)) as f:
        txt_list[i]=txt_list[i]+'\t'+f.read().strip()

with open(tgt_path,'w') as f:
    f.write('\n'.join(txt_list))

