import numpy as np
import RankingModel as rm
import RelRankGraph as rrg
import BayesNet as bn


class RelRankModel(rm.RankingModel):
    '''
    Relative ranking model
    '''
    def __init__(self,n):
        super().__init__(n= n)
        self.rr= rrg.RelRankGraph(n= n)
        self.p= bn.BayesNet(n= n, r= self.rr.card())


    def learn(self, rankings, args):
        '''
        Learn a realative ranking model from a set of full rankings

        args[0]: Dictionary with the parameters for learning the relative ranking graph
        args[1]: Dictionary with the paraemters for learning the Bayesian network over the relative ranking vars
        '''

        self.learn_rel_rank(rankings, args[0])
        self.learn_bayes_net(rankings, args[1])

    def learn_rel_rank(self, rankings, args):
        '''
        Learn a relative ranking graph from a set of full rankings

        args: dictionary with the parameters of the learning algorithm
        '''

        self.rr.learn(rankings, args)

    def learn_bayes_net(self, rankings, args):
        '''
        Learn a Bayesian network self.p over the relative ranking variables defined by self.rr from a set of rull
        rankings

        argsL dictionary with the parameters of the learning algorithm
        '''

        card= self.rr.card()
        data= self.rr.rank_to_relrank(rankings)

        self.p.learn(card, data, args)

