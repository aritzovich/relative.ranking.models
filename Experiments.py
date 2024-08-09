'''
Created on Sep 17, 2018

@author: aritz
'''

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets as datasets
import sklearn.cluster as cluster

import BayesNet as bnt
import Utils as tls
import RelRankGraph as rrg
import RelRankModel as rrm


def test_random_model(min_k= 1, max_k= 2, seed= 0):
    print("Learning random DG:")

    n=5
    m= 10
    np.random.seed(seed)

    # random rr model
    rand_model= rrm.RelRankModel(n)
    rand_model.randomModel(args= [min_k, max_k])

    tls.plot_graph_from_adjacency_list(Pa= rand_model.rr.Pa,title="Rel. Rank. Graph", file_name='rand_rrg.png')
    tls.plot_graph_from_adjacency_list(Pa= rand_model.rr.Pa,title="Factoriz. Graph", file_name='rand_fg.png')

    # sample rr model
    D= rand_model.sample_rr(m)



def NIPS22_previo(seed=1):
    #APA
    D = np.array(np.genfromtxt('APA_full.csv', delimiter=',', skip_header=1), dtype=np.int)
    votes= D[:,0]
    D= np.row_stack([np.row_stack([D[i,1:]-1 for j in range(votes[i])]) for i in range(D.shape[0])])
    N, n = D.shape

    Pa= rvg.learnVarTree(D)
    V= rvg.RVG(n,Pa)

    X= V.getDx(D)
    Tree= tls.maxLikelihoodTree(X)

    # Sushi is given in terms of ordering
    D = np.array(np.genfromtxt('sushi.csv', delimiter=','), dtype=np.int) - 1
    D = np.argsort(D, axis= 1)
    D = D[:, 1:]
    N, n = D.shape

    Pa,w= rrg.learnTree(D)
    V= rrg.RelRankGraph(n,Pa)
    Hy= np.zeros(100)
    for ind in range(100):
        randPa= tls.randomTree(n,ind)#[[]]+[[i] for i in range(0,n-1)]
        W= rrg.RelRankGraph(n= n, Pa= randPa)
        Y= W.ranking_to_relrank(D)
        y= Y[:,1]
        for i in range(2,n):
            y+= y*2+Y[:,i]
        cardY= len(np.unique(y))
        py= np.unique(y,return_counts=True)[1]/N
        Hy[ind]= np.sum(-py*np.log2(py))

    X= V.getDx(D)
    x= X[:,1]
    for i in range(2,n):
        x+= x*2+X[:,i]
    cardX= len(np.unique(x))
    Hx= np.unique(x,return_counts=True)[1]/N
    Hx= np.sum(-Hx*np.log2(Hx))
    Tree= tls.maxLikelihoodTree(X)

    data = np.array(np.genfromtxt('nascar2002.txt', delimiter=' ', skip_header=1), dtype=np.int) - 1
    # Load data: DriverID, Race, Place
    races = list(np.unique(data[:, 1]))
    driver = list(np.unique(data[:, 0]))
    D = list()
    for ind_race in range(len(races)):
        D.append(data[data[:, 1] == races[ind_race], 0][np.argsort(data[data[:, 1] == races[ind_race], 2])])
    data = D

    return


if __name__ == '__main__':
    test_random_model()
