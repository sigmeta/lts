

#for line in open('lts.test'):
#  line = line.strip().split(' ')
#  print(' '.join(list(line[0])) + '\t' + ' '.join(line[1:]))




src_dict = {}
tgt_dict = {}
for line in open('lts.s2s.train'):
  line = line.strip().split('\t')
  for word in line[0].split(' '):
    src_dict[word] = src_dict.get(word, 0)+1 

  for word in line[1].split(' '):
    tgt_dict[word] = tgt_dict.get(word, 0)+1 


#src_dict_list = sorted(src_dict.items(), key=lambda kv: kv[1], reverse=True)
#for k, v in src_dict_list:
#  print(k, v)

tgt_dict_list = sorted(tgt_dict.items(), key=lambda kv: kv[1], reverse=True)
for k, v in tgt_dict_list:
  print(k, v)


