import numpy as np

def greedy_match(S):
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  
    col = np.zeros((min_size))  

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result


def get_statistics(alignment_matrix, groundtruth_matrix,gth):
    #print(np.sum(groundtruth_matrix))
    
    greedy_pred = greedy_match(alignment_matrix)
    greedy_match_acc = compute_accuracy(greedy_pred, groundtruth_matrix)

    total = len(gth)
    hits_1, hits_5 = 0,0
    for k,v in gth.items():
        rk = np.sum(alignment_matrix[k,:]>alignment_matrix[k,v])+1
        if rk<=1:
            hits_1+=1
        if rk<=5:
            hits_5+=1

    h_1 = hits_1/total
    h_5 = hits_5/total

    return greedy_match_acc,h_1,h_5

def compute_accuracy(greedy_matched, gt):
    n_matched = 0
    for i in range(greedy_matched.shape[0]):
        if greedy_matched[i].sum() > 0 and np.array_equal(greedy_matched[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes
