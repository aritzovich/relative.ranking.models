import numpy as np
from scipy import special as scsp
import scipy.stats as stt
from itertools import combinations
import itertools as itr
from RankingModel import RankingModel


class Multistage(RankingModel):
    '''
    Multistage ranking model
    
    pref: "central" ordering/preferred ordering/error choices, np.array(int) of size n
    p: marginal distributions associated to each stage, list(np.array(float))
    prefInv: central ranking (list of integeres). 
        Given a set of elements S, prefInv[S] returns the indices of S in pref
    n: number of elements, {0,...,n-1}    
    '''

    def __init__(self, n):
        '''
        Constructor
        
        n: number of elements
        '''
        self.n = n
        self.pref= None

    def setPref(self, pref):
        self.pref = pref
        self.prefInv = np.argsort(pref)

    def randomModel(self, alpha=1.0, unimodal=True, seed= None):
        '''
        Samples choice probability distributions with non informative priors, i.e. alpha_s,v= alpha/(n-s+1)
        alpha: equivalent sample size
        '''
        if seed is not None:
            np.random.seed(seed)

        self.setPref(np.random.permutation(self.n))
        self.p = getRandomMarginals(self.n, alpha, unimodal=unimodal)

    # LEARN

    def learnFromOrderings(self, orders, alpha=0.0, w=None, pref=None):
        '''
        orders: list of orderings, np.array(int) num.orders X num.elems
        alpha: hyperparameter of the Dirichlet used as a priori
        '''

        if pref is None:
            if self.pref is None:
                self.setPref(consensusOrdering(orders))
        else:
            self.setPref(pref)

        self.p = getMarginalsFromOrderings(orders, self.pref, alpha, w)

    def learnFromTopOrderings(self, orders, alpha=0.0, w=None, pref=None):
        '''
        orders: a list of orderings list(list(int))
        alpha: hyperparameter of the Dirichlet used as a priori 
        '''
        if pref is None:
            if self.pref is None:
                self.setPref(consensusTopOrdering(self.n, orders))
        else:
            self.setPref(pref)

        self.p = getMarginalsFromTopOrderings(orders, self.pref, alpha, w)

    def learnFromSets(self, sets, alpha=0.0, w=None, pref=None):
        '''
        Learn the probability choices from sets (list(int))

        sets: a list of sets, (list(list[int]))
        alpha: hyperparameter of the Dirichlet used as a priori 
        ps: the marginals from a multistage model
        '''
        if pref is None:
            if self.pref is None:
                self.setPref(consensusOrderingSets(self.n, sets))
        else:
            self.setPref(pref)

        self.p = getMarginalsFromSets(self.pref.shape[0], sets, self.pref, alpha, w)

    # EVALUATE

    def eval(self, ord):
        return self.evals([ord])[0]

    def evals(self, ords):
        '''
        Evaluates the probability of an array full ordering

        ord: an array of orderings, np.array(int) num.ords X self.n
        '''
        error = orderingsToErrors(ords, self.pref)
        prob = np.ones(len(ords), dtype=np.float)
        for i, e in enumerate(error):
            for j in range(e.shape[0]):
                prob[i] *= self.p[j][e[j]]

        return prob

    def evalTop(self, top):
        return self.evalTops([top])[0]

    def evalTops(self, tops):
        '''
        Evaluates the probability of a list of top ordering, i.e.the sum of the probabilities of all the compatible orderings

        tops: a list of top orderings, list(list(int)) or list(np.array(int)), num.tops X len(tops[i])
        '''
        error = topOrderingsToErrors(tops, self.pref)
        prob = np.ones(len(tops), dtype=np.float)
        for i, e in enumerate(error):
            for j in range(e.shape[0]):
                prob[i] *= self.p[j][e[j]]

        return prob

    def evalSet(self, set, sample_size=1024, seed=None):
        '''
        Estimates the probability of a set, i.e., the sum of the probabilities of all the compatible orderings, by using
        a sample of sample_size compatible top orderings taken uniformly at random. When the number of compatible orderings
        is smaller than sample_size it computes the exact probability of the set
        '''

        if seed is not None:
            np.random.seed(seed)

        k = len(set)
        if 2 ** k <= sample_size:
            # Exact computation
            p = self.exactEvalSet(set)
        else:
            # Approximate computation
            p = np.average(self.evalTops([np.random.permutation(set) for _ in range(sample_size)]))
            p *= scsp.factorial(k)

        return p

    def exactEvalSet(self, S, return_subsets=False, pR=None):
        '''
        Obtains p(S)=sum_sigma p(sigma), where the sum is over all the orderings compatible with S. The computation is
        efficently performed in O(2^|S|) steps using a lattice representation over all subsets of S and the next
        recursion:

        p(S)= sum_{e in S} p_{S - e}(e) · p(S-e),

        where p_{S - e}(e) represents the choice set distribution of (general) L-decomposable models.


        :param S: set, list(int)
        :return:
        '''

        if not isinstance(S,list):
            S= list(S)

        if pR is None:
            pR = {R: 0 for k in range(len(S) + 1) for R in combinations(S, k)}
        else:
            pR.update({R: 0 for k in range(len(S) + 1) for R in combinations(S, k) if R not in pR})

        pR = {R: 0 for k in range(len(S) + 1) for R in combinations(S, k)}

        pR[()] = 1.0

        pS = self._dynProgEval(S, pR)

        if return_subsets:
            return pR

        return pS

    def _dynProgEval(self, S, pR):
        '''
        dynamic programming for the exact computation of the probability of a set
        '''

        key = tuple(S)
        if pR[key] == 0:
            s = len(S) - 1
            for i, e in enumerate(S):
                R = S[:i] + S[i + 1:]

                pR[key] += self.p[s][self.prefInv[e] - np.sum(self.prefInv[R] < self.prefInv[e])] * self._dynProgEval(R,
                                                                                                                      pR)

        return pR[key]

    def evalStages(self, order):
        '''
        Obtains the list of probabilities for each stage of a list of orderings,
        list(np.array(int)), with an ordering for each row
        '''
        error = orderingsToErrors(order, self.pref)
        eval = np.zeros(error.shape, dtype=np.float)
        for s in range(len(self.p)):
            eval[:, s] = self.p[s][error[:, s]]

        return eval

    def logLikelihood(self, orders):
        return np.sum(np.log2(self.eval(orders)), axis=0)

    def logLikelihoodStages(self, orders, norm=True):
        '''
        Log likelihood of each stage
        
        orders: permutations
        norm: normalize each stage by log2(n-s+1) for s=1,...,n-1
        '''
        if norm:  # Incluir la normalizacion
            return np.sum(np.log2(self.evalStages(orders)), axis=0) / (
                        np.array([np.log2(float(self.n - s)) for s in range(self.n)]) * orders.shape[0])
        else:
            return np.sum(np.log2(self.evalStages(orders)), axis=0)

    def divergence(self, ms_model):
        return [stt.entropy(self.p[s], ms_model.p[s], 2.0) / np.log2(float(self.n - s + 1)) for s in range(self.n)]

    # SAMPLE

    def sample(self, N=1):
        '''
        Sample N random orderings
        '''
        samples = np.zeros(shape=(N, len(self.p)), dtype=np.int)
        for x, p in enumerate(self.p):
            samples[:, x] = np.random.choice(len(self.p) - x, size=N, p=p)

        return errorsToOrderings(samples, self.pref)

    def sampleTop(self, N=1, q=None):
        '''
        Samples i.i.d top orderings or sets (of size at least 1) according to the multistage distribution.
        
        N: number of sets to be sampled
        q: distribution over the size of the sets to be sampled
        '''
        n = len(self.pref)
        if q is None:
            q = np.ones(n - 1, dtype=np.float) / (n - 1)

        top = [np.zeros(1 + np.random.choice(a=len(q), size=1, p=q), dtype=np.int) for i in range(N)]
        for i in range(N):
            for x in range(top[i].shape[0]):
                top[i][x] = np.random.choice(n - x, size=1, p=self.p[x])

        return errorsToSets(n, top, self.pref)

    # INFERENCE

    def getProbablestOrdering(self):
        '''
        Returns the most probable ordering.
        '''
        n = self.pref.shape[0]
        errors = np.array([np.argmax(pi) for pi in self.p])
        errors.shape = (1, n)
        return errorsToOrderings(errors, self.pref)

    def getProbablestCompatibleOrdering(self, partOrd):
        '''
        Returns the most probable total ordering compatible with a given (top) partial ordering
        
        partOrd: the partial ordering, np.array(int)
        '''
        error = np.zeros(self.n, dtype=np.int)
        error[np.arange(partOrd.size)] = partialOrderingsToErrors([partOrd], self.pref)
        for s in range(partOrd.size, self.n):
            error[s] = np.argmax(self.p[s])

        return errorsToOrderings(error, self.pref)

    def getExpectedCompatibleOrdering(self, set):
        # First part
        # ordered remainder codes to be used in the first len(set) positions
        remCodes = np.copy(np.sort(self.prefInv[set]))

        error = np.zeros(self.n, dtype=np.int)
        for s in np.arange(set.size):
            arg = np.argmax(self.p[s][remCodes])
            error[s] = remCodes[arg]
            # Actualize remainder error codes
            remCodes = np.delete(remCodes, arg)
            for i in np.arange(arg, remCodes.size):
                remCodes[i] -= 1

        # Second part
        for s in np.arange(set.size, self.n):
            error[s] = np.argmax(self.p[s])

        return errorsToOrderings(error, self.pref)

    def getProbablestSetCompatibleOrdering(self, set):
        '''
        Returns the most probable partial ordering compatible with a given choice set
        
        return the most probable compatible partial ordering, np.array(int)
        '''

        bestOrd = None
        highest = -np.Inf
        # Check among all the compatible partial ordering the most probable
        partOrds = [partOrd for partOrd in itr.permutations(list(set))]
        prob = self.evalPartial(partOrds)

        return partOrds[np.argmax(prob)]

    def getApproxProbablestSet(self, m):
        '''
        Obtain the approximated most probable set of size m
        
        m: the size of the set
        '''
        errors = np.array([np.argmax(self.p) for i in range(m)])
        errors.shape = (1, m)
        return errorsToSets(n=m, errors=errors, pref=self.pref)

    def getProbablestOrderings(self):
        '''
        Get the most probable ordering
        '''
        n = len(self.p)
        errors = [np.argsort(self.p[s]) for s in range(n - 1)]
        return errorsToOrderings(errors, self.pref)



'''
######################
# CONSENSUS ORDERING #
######################
'''


def consensusOrdering(order, w=None):
    '''
    weighted Borda ordering: Given a list of total orderings obtain the ordering induced by the the average
    ranking of each element

    order: list of total orderings (list of list of integers)
    '''

    if w is None:
        w = np.ones(len(order))

    ranking = np.sum(np.argsort(np.array(order), axis=1) * w.reshape((w.shape[0], 1)), axis=0)

    # Obtain the ordering associated to the average ranking
    return np.argsort(ranking)


def consensusTopOrdering(n, order, w=None):
    '''
    Given a list of total orderings obtain the ordering induced by the the average
    ranking of each element

    Recuento Borda?

    order: list of partial top-k orderings with k variable (list of list of integers)
    '''

    N = len(order)

    if w is None:
        w = np.array([i for i in range(n)])
    avgRanking = np.zeros((n,), dtype=np.float)
    for i in range(N):
        ordering = order[i];
        # add the ranking associate to the ordering
        avgRanking[:len(ordering)] += w[np.argsort(ordering)]
        avgRanking[len(ordering):] = (n - len(ordering)) / 2

    # Obtain the ordering associated to the average ranking
    return np.argsort(avgRanking)


'''
Condorcet's reference ordering
'''


def consensusCondorcet(order):
    N = len(order)
    n = len(order[0])
    rankings = np.argsort(order)
    wins = np.zeros(n)
    for r in range(N):
        for i in range(n):
            for s in range(r + 1, N):
                if (rankings[r][i] > rankings[s][i]):
                    wins[i] += (n - rankings[r][i])
                else:
                    wins[i] += (n - rankings[s][i] - 1)

    return np.argsort(-wins)


# @profile
def consensusOrderingSets(n, sets, w=None):
    '''
    Given a list of partial ranks expressed in terms of sets, i.e. the elements
    in the set has lower rank than the elements outside the set, computes the ordering
    induced by the average ranking of the elements

    n: size of the rankings
    sets: list of sets (set: list of integers)
    w: wights of the sets
    '''
    m = len(sets)
    if w is None:
        w = np.ones(m)

    ranking = np.zeros((n,), dtype=np.float)
    for iS, S in enumerate(sets):
        ranking += w[iS] * (n - 1 + len(S)) / 2.0
        ranking[S] += w[iS] * (len(S) - 1) / 2.0 - (n - 1 + len(S)) / 2.0

    return np.argsort(ranking)


def consensusPartOrdering(n, l, w=None):
    '''
    Borda ordering with top-k partial orderings
    :param n: total number of elements
    :param l: a list of total orderings, list(np.array(int))
    :return: the central Borda ordering
    '''

    m = len(l)
    if w is None:
        w = np.ones(m)

    # Sum of ranking associated to a list of empty sets
    ranking = np.zeros((n,), dtype=np.float)
    for iS, S in enumerate(l):
        ranking += w[iS] * (n - 1 + len(S)) / 2.0
        ranking[S] += w[iS] * (np.arange(len(S)) - (n - 1 + len(S)) / 2.0)

    return np.argsort(ranking)


'''
#################
# CODIFICATIONS #
#################
'''


def orderingsToErrors(order, pref=None):
    '''
    Transforms a an arrau of total ordering into its codification given in terms of
    discords/error choices assuming that the preferences between elements are given by
    the ordering.

    order: array of total orderings (np.array(shape=(numOrderings,numElems),int)
    pref: central ordering/preferences over elements, np.array(int)

    return: array of error choices, np.array(int) num.orders X self.n
    '''
    if isinstance(order,list):
        order= np.array(order)

    n = order.shape[1]

    if pref is None:
        pref = np.arange(n)

    pos = np.argsort(pref)
    errors = pos[order[:, ]]
    corr = np.zeros(errors.shape, dtype=np.int)
    for i in np.arange(n - 1):
        for j in np.arange(i + 1, n):
            corr[:, j] -= (errors[:, i] < errors[:, j])

    errors += corr

    return errors


def topOrderingsToErrors(orders, pref=None):
    '''
    Transforms a list of partial orderings into a list of their codifications given in
    terms of discords/error choices assuming that the preferences between elements are
    given by the pref ordering.
    
    orders: list of partial orderings list(np.array(int)) or list(list(int))
    pref: central ordering/preferences over elements, np.array(int)
    
    return a list of discords/error choices list(np.array(int))
    
    CHECK 2017.11.13
    '''
    if pref is None:
        pref = np.arange(np.max([o.size for o in orders]))

    pos = np.argsort(pref)
    errors = list()
    for o in orders:
        try:
            e = pos[o]
        except:
            e = pos[o]

        try:
            corr = np.zeros(o.size, dtype=np.int)
        except:
            corr = np.zeros(o.size, dtype=np.int)

        for i in np.arange(o.size - 1):
            for j in np.arange(i + 1, o.size):
                corr[j] -= (e[i] < e[j])

        e += corr

        errors.append(e)

    return errors;


def errorsToOrderings(error, pref=None):
    '''
    Transforms a vector of error choices/discords into a total ordering of elements
    assuming that the preferences between elements are given by the ordering 
    given in refer. If pref==None 1,...,n is taken as the preferred ordering.
    
    order: list of errors wrt the preferences (np.array(shape=(numOrderings,numElems),int)
    pref: central ordering/preferences over elements, np.array(int)
    '''
    if len(error.shape) == 2:
        (N, n) = error.shape
        errors = error
    else:
        N = 1
        n = error.shape[0]
        errors = np.copy(error)
        errors.shape = (N, n)

    if pref is None:
        pref = np.arange(n)

    order = np.zeros((N, n), dtype=np.int)
    rem = [[j for j in np.arange(n)] for i in np.arange(N)]
    for j in np.arange(n):
        for i in np.arange(N):
            order[i, j] = pref[rem[i][errors[i, j]]]
            del rem[i][errors[i, j]]

    if len(error.shape) == 2:
        return order
    else:
        return order[0, :]


def errorsToSets(n, errors, pref=None):
    N = len(errors)
    if pref is None:
        pref = np.arange(n)

    sets = list()
    for i in range(N):
        rem = list(pref)
        sets.append(np.zeros(errors[i].shape[0], dtype=np.int))
        for j in range(errors[i].shape[0]):
            sets[i][j] = rem[errors[i][j]]
            del rem[errors[i][j]]

    return sets


def errorsToTopOrderings(n, errors, pref=None):
    N = len(errors)
    if pref is None:
        pref = np.arange(n)

    topOrd = list()
    for i in range(N):
        rem = list(pref)
        topOrd.append(np.zeros(errors[i].shape[0], dtype=np.int))
        for j in range(errors[i].shape[0]):
            topOrd[i][j] = rem[errors[i][j]]
            del rem[errors[i][j]]

    return topOrd


def indexeToErrors(ind, n, pref=None):
    '''
    Given an index (interger in {0,...,n!-1}) returns an unique vector of error choices 
    (the vector of error choices is interpreted as a number in base (n,n-1,...,1) and 
    the weight is lower for the errors choices at lower positions in the vector.
    
    see errorsToIndexes
    
    error: error choices (list of integers)
    pref: preferred ordering
    '''
    if pref is None:
        pref = [i for i in range(n)]

    div = ind
    errors = list()
    for i in range(n):
        rem = div % (n - i)
        errors.append(rem)
        div = div % (n - i)

    return errors


def errorsToIndex(error, pref=None):
    '''
    Given a vector of error choices returns an unique index (the vector of error choices
    is interpreted as a number in base (n,n-1,...,1) and the weight is lower for the
    errors choices at lower positions in the vector.
    
    see indexToErrors
    
    error: error choices (list of integers)
    pref: preferred ordering
    '''
    n = len(error)
    ind = 0
    w = 1
    for i in range(n):
        ind += error[i] * w
        w * (n - i)

    return ind


def getPositionsOf(refer, elems):
    '''
    Obtain the position of a list of elements in a reference list of elements, no.array
    '''
    return [np.where(refer == elem) for elem in elems]


'''
#######################
# EVALUATION MEASURES #
#######################
'''


def sumErrors(orders, pref=None):
    N = len(orders)
    n = len(orders[0])

    if pref is None:
        pref = [i for i in range(n)]

    sum = 0
    for i in range(N):
        sum += np.sum(orderingToErrors(orders[i], pref))

    return sum


def kunchevaIndex(A, B, n):
    '''
    Kuncheva's consistency index.

    :param A: a set of elements, np.array(int)
    :param B: a set of elements, np.array(int)
    :param n: total number of elements
    :return: Kuncheva's consistency index for sets A and B from a set of n elements
    '''

    A = set(A)
    B = set(B)
    k = len(A)
    r = len(A.intersection(B))
    return (r * n - k ** 2) / (k * (n - k))


'''
##############################
# PARTIAL ORDERING UTILITIES #
##############################
'''


def getPairwiseOrderings(ordering):
    '''
    Given a list of orderings (list of integers, o[i]: the element o[i] has rank i)
    obtains the matrix of pairwise partial rankings, R where R[i,j] indicates the number of times that
    element i has a lower ranking than element j in the list or rankings
    '''
    N = len(ordering)
    n = len(ordering)
    R = np.zeros((n, n))
    for o in ordering:
        for i in range(n):
            for j in range(i + 1, n):
                R[o[i], o[j]] += 1

    return R


def gerPairwiseRanking(ranking):
    '''
    Given a list of rankings (list of integers, r[i]: the position of element i given the rank r)
    obtains the matrix of pairwise partial rankings, R where R[i,j] indicates the number of times that
    element i has a lower ranking than element j in the list or rankings
    '''
    N = len(ranking)
    n = len(ranking)
    R = np.zeros((n, n))
    for r in ranking:
        for i in range(n):
            for j in range(i + 1, n):
                if (r[i] < r[j]):
                    R[i, j] += 1
                else:
                    R[j, i] += 1


def getPairwiseRankingFromSets(n, sets):
    '''
    Given a list of sets of elements representing partial ordering, it returns the average partial pairwise rankings by considering
    all the rankings compatible with the partial ordering given by the list of sets
    
    n: number of vars
    sets: list of sets (list(list[int]) )
    
    return: np.array(nxn) of pairwise rankings
    '''
    R = np.zeros(n, n)
    for S in sets:
        higher = [i for i in range(n) if i not in S]
        for s in S:
            for h in higher:
                R[s, h] += 1

    return None


def getRandomSetCompatibleOrdering(sets):
    '''
    Obtains a list of partial orderings compatible with a given list of sets
    
    selts: list of choice sets, List(np.array(int))
    '''
    return [np.random.permutation(S) for S in sets]


def getRandomPartialCompatibleOrdering(partOrds, n):
    '''
    Obtains a list of total orderings compatible with a given list of partial orderings
    
    partOrds: list of partial orderings, List(np.array(int))
    n: number of elements, int
    
    return list of total orderings, np.array(int) with a total ordering for each row
    '''

    elems = np.arange(n)
    ords = np.array(shape=(len(partOrds), n))
    for i in np.arange(len(partOrds)):
        ords[i, :] = np.concatenate(partOrds[i], np.random.permutation(np.setdiff1d(elems, np.sort(partOrds[i]))))

    return ords


def getRandomTopOrders(pi, m, k, d_min=1, d_max=None):
    '''
    Generates m random orderings at (Kendall's tau) distance in the range [d_min,d_max] (when possible) from pi in their top-k positions

    Improvement: guarantee the bounds for the minimum and maximum distance

    :param pi: reference ordeing
    :param k: k, of the top-k orderings
    :param m: number of top-k orderings
    :param d_min: minimum distance
    :param d_max: maximum distance
    :return:
    '''

    n = len(pi)

    errors = np.zeros((m, n), dtype=np.int)
    for i in range(m):
        s, e = np.unique(np.random.choice(k, np.random.randint(d_min, d_max, size=1)), return_counts=True)
        errors[i, s] = e

    for j in range(k):
        errors[errors[:, j] > n - j - 1, j] = n - j - 1

    sigma = errorsToOrdering(errors, pi)

    return sigma


'''
#################################
# LEARNING CHOICE DISTRIBUTIONS #
#################################
'''


def getMarginalsFromOrderings(order, pref=None, alpha=0.0, w=None):
    '''
    Given a list of total orderings and a central ordering obtains the marginals of 
    the error choices variables associated to each stage of the multistage model
    
    order: list of total orderings (list of list of integers)
    pref: central ordering/preferences over elements (list of integers)
    alpha: parameters of the prior Dirichlet for the marginal of each stage. Note that the equivalent sample size of
    the priors for stage s alpha*(n-s+1)
    w: probability mass of each ordering. If None, the weight is 1
    OPTIMIZAR: trabajar con numpy exclusivamente: matrices 
    '''
    if type(order) is list:
        N = len(order)
        n = order[0].size
    else:
        (N, n) = order.shape

    if pref is None:
        pref = np.arange(n)

    if w is None:
        w = np.ones(N)

    p = [np.ones((n - i,), dtype=np.float) * alpha for i in np.arange(n)]
    errors = orderingsToErrors(order, pref)

    for i in np.arange(n):
        inds = np.unique(errors[:, i])
        for ind in inds:
            try:
                p[i][ind] += np.sum((errors[:, i] == ind) * w)
            except:
                p[i][ind] += np.sum((errors[:, i] == ind) * w)

        N = np.sum(w)

    for s in np.arange(len(p)):
        p[s] /= (N + alpha* len(p[s]))

    return p


def getMarginalsFromTopOrderings(order, pref=None, alpha=0.0, w=None, criteria=0):
    '''
    Given a list of total orderings and a central ordering obtains the marginals of 
    the error choices variables associated to each stage of the multistage model
    
    order: list of partial orderings (list of list of integers)
    pref: central ordering/preferences over elements (list of integers)
    alpha: parameters of the prior Dirichlet for the marginal of each stage. Note that the equivalent sample size of
    the priors for stage s alpha*(n-s+1)
    criteria: learning criteria (impact on the efficiency)
    - 0: A parcial ordering, ord, only modifies the parameters of the first |ord| stages
    - 1: A partial ordering modifies the parameters of all the stages
    '''

    N = np.zeros(len(pref))

    if pref is None:
        n = np.max([e.size for e in order])
        pref = np.arange(n)
    else:
        n = len(pref)

    if w == None:
        w = np.ones(len(order))

    p = [np.ones((n - i,), dtype=np.float) * alpha for i in np.arange(n)]
    errors = topOrderingsToErrors(order, pref)

    for i, e in enumerate(errors):
        for s in range(len(e)):
            p[s][e[s]] += w[i]
            N[s] += w[i]

        if criteria == 1:
            for s in range(len(e), n):
                p[s] += w[i] * np.ones(n - s) / np.float(n - s)
                N[s] += w[i]

    for s in range(len(p)):
        p[s]/= (N[s] + alpha * len(p[s]))

    return p


def getMarginalsFromSets(n, sets, pref=None, alpha=0.0, w=None, criteria=0):
    '''
    Learn the the parameters of the choice distributions from choice sets.
    The estimation takes into account all the orderings compatible with a choice set
    giving the same weight to any of them (a weight of 1/number of compatible orderings) 
    By setting alpha>0 Bayesian learning of the parameters is performed.
    
    :param sets: list of sets, list(np.array(int)), representing preference sets
    pref: central ordering/preferences over elements (list of integers)
    :param alpha: parameters of the prior Dirichlet for the marginal of each stage. Note that the equivalent sample size of
    the priors for stage s alpha*(n-s+1)
    :param criteria: learning criteria
    - 0: A parcial ordering, ord, only modifies the parameters of the first |ord| stages
    - 1: A partial ordering modifies the parameters of all the stages

    return the choice distributions
    '''

    if pref is None:
        pref = np.arange(n)

    if w is None:
        w = np.ones(len(sets))

    centralRank = np.argsort(pref)
    sortedRank = [np.sort(centralRank[S]) for S in sets]

    p = [np.ones((n - i,), dtype=np.float) * alpha for i in np.arange(n)]

    N = np.zeros(n)
    for iS, S in enumerate(sortedRank):
        t = S.shape[0]
        for ind in np.arange(t):  # All the elements
            r = S[ind]  # t^-=ind, t^+=m-ind-1, ind is the number of elements with lower rank in S
            for s in np.arange(t):  # Choice probs of stage 1...t
                for j in np.arange(np.max([0, s - t + ind + 1]),
                                   np.min([ind, s]) + 1):  # all the possible values: {min(m^-,i),...,max(0,i-m^+)}
                    p[s][r - j] += __combinatorial(t - ind - 1, ind, s, j) * w[iS]

            N[ind] += w[iS]

        if criteria == 1:
            for s in np.arange(t, n):
                p[s] += np.ones(n - s, dtype=np.float) / (n - s) * w[iS]
                N[s] += w[iS]

    for s in range(n):
        p[s] /= (alpha * len(p[s]) + N[s])

    return p


def getApproxMarginalsFromSets(n, sets, pref=None, alpha=0.0, criteria=0, numSamples=100):
    '''
    Learn the APPROXIMATE parameters of the choice distributions from choice sets.
    The estimation takes into account sampleSize the orderings compatible with a choice set
    giving the same weight to any of them (a weight of 1/number of compatible orderings)
    By setting alpha>0 Bayesian learning of the parameters is performed.

    sets: list of sets, list(np.array(int)), representing preference sets
    pref: central ordering/preferences over elements (list of integers)
    :param alpha: parameters of the prior Dirichlet for the marginal of each stage. Since we are replicating each set
    numSamples times, alpha is multiplicated by num samples for maintaining the proportion w.r.t. the number of sets
    Note that the equivalent sample size of the priors for stage s alpha*(n-s+1)*numSamples
    :param criteria: learning criteria
    -0: A partial ordering modifies the parameters of all the stages
    -1: A parcial ordering, ord, only modifies the parameters of the first |ord| stages

    return the choice distributions
    '''

    if pref is None:
        pref = np.arange(n)

    compOrd = list()
    for S in sets:
        compOrd.extend([np.random.permutation(S) for i in range(numSamples)])

    return getMarginalsFromTopOrderings(compOrd, pref, alpha=numSamples * alpha)


def __combinatorial(plus, minus, i, j):
    val = (scsp.binom(minus, j) / scsp.binom(minus + plus, i)) * (scsp.binom(plus, i - j) / (minus + plus + 1))
    return val


'''
Generates the marginal distributions by sampling a Dirichlet with uniform parameters alpha_s,v= alpha

Strongly unimodal multistage with central permutation equal to reference permutation when alpha[i]>alpha[i+1] for i=0,..,n-2
'''


def getRandomMarginals(n, alpha, unimodal=True):

    if unimodal:
        return [-np.sort(-np.random.dirichlet(np.ones(n - s) * alpha)) for s in range(n)]
    else:
        return [np.random.dirichlet(np.ones(n - s) * alpha) for s in range(n)]


######################################
# AUXILIAR METHODS for SIZES of SETS #
######################################

'''
Learn the distribution of the sizes of the sets

n: number of elements that can be included in the sets
sets: list of sets (list[set(int)])
alpha: hyper-parameter, sample size
'''


def getSetSizeModel(n, minSize, maxSize, sets, alpha=0.0):
    p = np.concatenate((np.zeros(minSize),
                        np.ones(maxSize + 1 - minSize, np.float) * alpha / float(maxSize + 1 - minSize),
                        np.zeros(n - maxSize - 1)))
    for s in sets:
        p[len(s)] += 1

    p = p / float(len(sets) + alpha)

    return p


def toBinarySets(n, sets):
    bin = np.zeros(shape=(len(sets), n))
    for i, s in enumerate(sets):
        for j in s:
            bin[i, j] = 1

    return bin
