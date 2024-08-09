import numpy as np
import Utils as tls

class RelRankGraph(object):
    '''
    Relative ranking graph. The directed acyclic graph that define the relative ranking variables that conform the
    relative ranking model
    '''
    def __init__(self,n,Pa=None):
        '''
        Pa: list of parents, list(list(int)), Pa[i]: the parents of i
        '''
        self.n= n

        if Pa is None:
            self.Pa= list(np.array([]) for i in range(n))
        else:
            self.Pa= [pa.copy() for pa in Pa]

    def card(self):
        '''
        Get the cardinality of the relative ranking variables
        '''

        return np.array([len(self.Pa[i]) + 1 for i in range(self.n)])

    def addPa(self,Pa):
        '''
        Update the set of parents by adding Pa
        Pa: dictionary, index:list(parents)
        '''
        for i in Pa.keys:
            self.Pa[i].update(Pa[i])

    def removePa(self,Pa):
        '''
        Update the set of parents by removing Pa
        Pa: dictionary, index:list(parents)
        '''
        for i in Pa.keys:
            self.Pa[i]= self.Pa[i].difference(set(Pa[i]))

    def ranking_to_relrank(self,rankings):
        '''
        Transform a ranking into a configuration of ranking vars
        '''

        if isinstance(rankings, np.array):
            relrank= np.zeros(self.n)
            for i in range(self.n):
                S= list(i)+ list(self.Pa[i])
                relrank[i]= np.argsort(rankings[S])[0]

        if isinstance(rankings, np.matrix):
            m= rankings.shape[0]
            relrank= np.zeros(shape=(m,self.n),dtype=np.int)
            for i in range(self.n):
                S= [i]+ list(self.Pa[i])
                relrank[:,i]= np.argsort(rankings[:,S],axis=1)[:,0]

        return relrank

    def random(self, args=[1,1], seed= None):

        if seed is not None:
            np.random.seed(seed)

        min_k= args[0]
        max_k= args[1]

        self.Pa= tls.randomDecGraph(n= self.n, minK= min_k, maxK= max_k)[1]


    def learn(self, rankings, args= None):
        '''
        Learn the relative ranking graph from a training set of rankings

        Implemented: maximum likelihood tree
        '''

        D= self.ranking_to_relrank(rankings)
        self.Pa, w= tls.maxLikelihoodTree(D= D)

        return w


def learnTree(D):
    '''
    Return the relative ranking graph with tree structure that maximizes the entropy of the relative ranking
    model with the variables defined by the tree and the factorization associated to the empty graph
    '''

    (m,n)= D.shape

    # Compute the entropies associated to sigma(i)<sigma(j), sigma_ij(i), sigma_i|j
    W = np.zeros(shape=(n, n))
    for i in range(n):
        W[i,i] = 0
        for j in range(i+1,n):
            p = np.unique(D[:, i]<D[:, j], return_counts=True)[1]/m
            W[i, j] = -np.sum(p*np.log2(p))
            W[j, i] = W[i, j]

    return tls.maximumWeigtedTree(W)[0]
