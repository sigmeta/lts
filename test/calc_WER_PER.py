import sys

def print_rate(prompt, error_count, total_count):
    print(prompt, error_count, '/', total_count, '=', error_count / total_count)

def calc_edit_distance(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)
    f = [[0] * (len2 + 1) for i in range(len1 + 1)]
    for i in range(1, len1 + 1):
        f[i][0] = i
    for j in range(1, len2 + 1):
        f[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                f[i][j] = f[i-1][j-1]
            else:
                f[i][j] = min(f[i-1][j], f[i][j-1], f[i-1][j-1]) + 1
    return f[len1][len2]

def compare(reference, inference):
    assert len(reference) == len(inference)

    token_count = 0
    dist_sum = 0
    error_count = 0
    for i in range(len(reference)):
        dist = calc_edit_distance(reference[i], inference[i])
        dist_sum += dist
        token_count += len(reference[i])
        if dist != 0:
            error_count += 1
    
    print_rate('WER:', error_count, len(reference))
    print_rate('PER:', dist_sum, token_count)

def read_sequences(path):
    seqs = []
    with open(path, 'r') as f:
        for line in f.readlines():
            seqs.append(line.strip().split())
    return seqs

def remove_stress(seqs):
    for seq in seqs:
        for i in range(len(seq)):
            seq[i] = seq[i].rstrip('12')

assert len(sys.argv) == 3, sys.argv

if __name__ == '__main__':
    reference = read_sequences(sys.argv[1])
    inference = read_sequences(sys.argv[2])
    compare(reference, inference)

    print('If remove stress:')
    remove_stress(reference)
    remove_stress(inference)
    compare(reference, inference)
