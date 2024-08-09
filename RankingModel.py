import numpy as np
from scipy import special as scsp
import itertools as itr
import RelRankGraph as rrg
import BayesNet as bn


class RankingModel(object):
    '''
    Abstract class of ranking models. It gives implementation to the full ranking model
    '''

    def __init__(self, n):
        self.n= n

    def randomModel(self, args= [1,1,1,1,1.0], seed= None):

        if seed is not None:
            np.random.seed(seed)

        self.rr= rrg.RelRankGraph(self.n)
        self.rr.random(args= args[:2])

        self.p= bn.BayesNet(n= self.n, r= self.rr.card())
        self.p.random(args[2],args[3],args[4])


    def learn(self, rankings, args= ["mhtree","mltree"]):
        self.rr= rrg.RelRankGraph(self.n)
        self.rr.learn(rankings,args=[args[0]])
        self.p= bn.BayesNet(n=self.n, r= self.rr.card())
        D= self.rr.ranking_to_relrank(rankings)
        self.p.learn(D,args=[args[1]])

    def evaluate(self, rankings):
        D= self.rr.ranking_to_relrank(rankings)
        self.p.evaluate(D)

    def sample(self, m):
        None

    def sample_relrank(self, m, seed= None):
        if seed is not None:
            np.random.seed(seed)

        D= self.p.sample(m= n)
        return D

