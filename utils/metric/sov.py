def split_segs(seq1, state=1):
    S1_1 = []
    S1_0 = []
    s, t = 0, 0
    while s < len(seq1):
        while t < len(seq1) and seq1[s] == seq1[t]:
            t += 1
        s1 = {'s': s, 't': t - 1, 'l': t - s}
        if str(seq1[s]) == str(state):
            S1_1.append(s1)
            s = t
        else:
            S1_0.append(s1)
            s = t
    return S1_1, S1_0


def get_overlap_set(S1, S2):
    S_overlap = []
    S_non_overlap = []
    for s1 in S1:
        flag = True
        for s2 in S2:
            if s1['t'] < s2['s'] or s1['s'] > s2['t']:
                pass
            else:
                flag = False
                S_overlap.append((s1, s2))
        if flag:
            S_non_overlap.append(s1)
    return S_overlap, S_non_overlap


def delta_score(delta=None, s1=None, s2=None):
    if delta is not None:
        return delta
    else:
        candidate = [maxov(s1, s2) - minov(s1, s2), minov(s1, s2), s1['l'] // 2, s2['l'] // 2]
        return min(candidate)


def minov(s1, s2):
    s = max(s1['s'], s2['s'])
    t = min(s1['t'], s2['t'])
    return t - s + 1


def maxov(s1, s2):
    s = min(s1['s'], s2['s'])
    t = max(s1['t'], s2['t'])
    return t - s + 1


def Sov_state(S_overlap, S_non_overlap, delta=None):
    S_i, N_i = 0, 0
    for s1, s2 in S_overlap:
        s = (minov(s1, s2) + delta_score(delta, s1, s2)) / maxov(s1, s2) * s1['l']
        S_i += s
        N_i += s1['l']
    for s1 in S_non_overlap:
        N_i += s1['l']
    Sov_i = S_i / N_i * 100
    return Sov_i, S_i, N_i


def Sov(observe, prediction, delta=None):
    S1_pos, S1_neg = split_segs(observe)
    S2_pos, S2_neg = split_segs(prediction)
    S_overlap_pos, S_non_overlap_pos = get_overlap_set(S1_pos, S2_pos)
    S_overlap_neg, S_non_overlap_neg = get_overlap_set(S1_neg, S2_neg)
    if len(S_overlap_pos) == len(S_non_overlap_pos) == 0:
        print('observe', observe)
        print('S1_pos', S1_pos)
        print('prediction', prediction)
        print('S2_pos', S2_pos)
    Sov_1, S_1, N_1 = Sov_state(S_overlap_pos, S_non_overlap_pos, delta)  # state=1
    Sov_0, S_0, N_0 = Sov_state(S_overlap_neg, S_non_overlap_neg, delta)  # state=0
    Sov = (S_1 + S_0) / (N_1 + N_0) * 100
    return Sov, Sov_1, Sov_0
