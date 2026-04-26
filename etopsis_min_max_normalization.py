###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank 
def ranking_e(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.title('ETOPSIS', loc='right')
    plt.show() 
    return

def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.title('TOPSIS', loc='right')
    plt.show() 
    return

# Function: MIN-MAX normalization of 1D vector

def min_max_normalize(v):
    return (v - v.min()) / (v.max() - v.min())

# Function: ETOPSIS

def etopsis_method(dataset, weights, epsilon, criterion_type, 
                   graph = True, verbose = True):
    X = np.copy(dataset)
    w = np.copy(weights)

    mean_w = w/max(w)

    m, n = X.shape
    US = np.zeros((m, n))

    for i in range(0, n):
        col = X[:, i]
        
        min_val = np.min(col)
        max_val = np.max(col)
        
        if max_val == min_val:
            US[:, i] = 0
        else:
            US[:, i] = (col - min_val) / (max_val - min_val)
        
        if criterion_type[i] == "min":
            US[:, i] = 1 - US[:, i]
        elif criterion_type[i] == "max":
            pass
        else:
            pass

    VS = US * mean_w    
    ranges = np.ptp(VS, axis=0)           
    ranges = ranges / np.max(ranges)     

    mw = np.mean(mean_w)                  
    nw = np.linalg.norm(mean_w)          
    s = nw / mw

    WAM = VS @ (mean_w  / nw)             # VS * (mean_w /nw)
    # WMSD = [wam, sum((VS - wam*(mean_w s/nw)).^2, 2).^0.5] / s
    diff = VS - np.outer(WAM, mean_w / nw)    # różnica VS - wam*(mean_w /nw)
    WSD = np.sqrt(np.sum(diff**2, axis=1))     # odchylenie (2-norma wiersza)
    WMSD = np.column_stack((WAM, WSD)) / s     # połączenie w jedną macierz i skalowanie 

    A = np.sqrt(WMSD[:, 0]**2 + (WMSD[:, 1] / epsilon)**2)
    I = np.sqrt((np.mean(mean_w) - WMSD[:, 0])**2 + (WMSD[:, 1] / epsilon)**2)
    R = A / (A + I)


    if (verbose == True):
        for i in range(0, R.shape[0]):
            print('a' + str(i+1) + ': ' + str(np.round(R[i], 2)))
    if ( graph == True):
        flow = np.copy(R)
        flow = np.reshape(flow, (R.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, R.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking_e(flow)

###############################################################################

# Function: TOPSIS
def topsis_method(dataset, weights, criterion_type, graph = True, verbose = True):
    X = np.copy(dataset)
    w = np.copy(weights)
    # sum_cols = np.sum(X*X, axis = 0)
    # sum_cols = sum_cols**(1/2)
    # r_ij = X/sum_cols
    # v_ij = r_ij*w
    m, n = X.shape
    v_ij = np.zeros((m, n))
    for i in range(0, n):
        col = X[:, i]
        col_norm = min_max_normalize(col)

        for j in range(m):
            v_ij[j, i] = col_norm[j]

    mean_w = w/max(w)
    v_ij = v_ij * mean_w

    p_ideal_A = np.zeros(X.shape[1])
    n_ideal_A = np.zeros(X.shape[1])
    for i in range(0, dataset.shape[1]):
        if (criterion_type[i] == 'max'):
            p_ideal_A[i] = np.max(v_ij[:, i])
            n_ideal_A[i] = np.min(v_ij[:, i])
        else:
            p_ideal_A[i] = np.min(v_ij[:, i])
            n_ideal_A[i] = np.max(v_ij[:, i])
    p_s_ij = (v_ij - p_ideal_A)**2
    p_s_ij = np.sum(p_s_ij, axis = 1)**(1/2)
    n_s_ij = (v_ij - n_ideal_A)**2
    n_s_ij = np.sum(n_s_ij, axis = 1)**(1/2)
    c_i    = n_s_ij / ( p_s_ij  + n_s_ij )
    if (verbose == True):
        for i in range(0, c_i.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(c_i[i], 2)))
    if ( graph == True):
        flow = np.copy(c_i)
        flow = np.reshape(flow, (c_i.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, c_i.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return c_i

###############################################################################

#Epsilon
epsilon = 1

# Weights
weights =  [0.6, 0.4, 0.8, 0.2]

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'min', 'max', 'min']

# Dataset
dataset = np.array([
                [6, 86, 4, 10],   #a1
                [9, 23, 34, 6],   #a2
                [24, 9, 7, 23],   #a3
                [8, 2, 75, 8],   #a4
                [4,89, 2, 3],   #a5
                [7, 50, 9, 92],   #a6
                [9, 5, 3, 91],   #a7
                [3, 45, 70, 6],   #a8
                [0, 0, 0, 0],   #a9
                [4, 6, 76, 9],   #a10
                ])

print("TOPSIS:")
TOPSIS_relative_closeness = topsis_method(dataset, weights, criterion_type, graph = True, verbose = True)

print("ETOPSIS:")
ETOPSIS_relative_closeness = etopsis_method(dataset, weights, epsilon, criterion_type, graph = True, verbose = True)