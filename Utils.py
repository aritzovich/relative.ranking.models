import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def randomDecGraph(n,minK,maxK,seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    remain= np.random.permutation(n)
    
    #random ancestral order
    anc=np.ndarray.tolist(np.fliplr([remain])[0])

    cliques=[]
    Pa=[[] for i in range(n)]
    #Create the first clique
    C=[remain.pop() for i in range(np.random.choice(maxK+1-minK)+minK)]
    for i in range(1,len(C)):
        Pa[C[i]]=[C[j] for j in range(i)]
    cliques.append(C)
    
    #Create the remaining cliques
    while remain:
        X= remain.pop()
        ancC= cliques[np.random.choice(len(cliques))]
        Pa[X]= np.ndarray.tolist(np.random.choice(ancC,np.random.choice(maxK-minK+1)+minK-1, replace=False))
        C=[X]
        C.extend(Pa[X])
        cliques.append(C)
    
    return (anc,Pa,cliques)


def fromEdgesToAdjacencyList(n,E):
    '''
    Given a list of edges List((u,v)) representing a decomposable graph obtains
    the adjacency list representation List(list(int)), where v in adj[u] if and 
    only if (u,v) in E
    '''
    
    adj= [list() for i in range(n)]
    
    for e in E:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
        
    return adj

def updateAdjacencyListFromEdges(E,dictA):
    '''
    Given a list of edges, (i,j), and a dictionary of adjacency list indexed by the 
    index of the nodes updates the dictionary with the new adjacences given by
    the list of edges.
    
    The method does not control duplicate adjacences
    '''
    
    for e in E:
        if dictA.has_key(e[0]):
            dictA[e[0]].append(e[1])
        else:
            dictA.update({e[0]:[e[1]]})

        if dictA.has_key(e[1]):
            dictA[e[1]].append(e[0])
        else:
            dictA.update({e[1]:[e[0]]})
    
    return dictA

def fromAdjacencyListToEdges(adj):
    E= list()
    for u in range(len(adj)):
        for v in adj[u]:
            E.append((u,v))
            
    return E

def fromAdjacencyListToParents(n,adj):
    '''
    Given an adjacency list representing a decomposable graph obtains the directed
    representation in terms of a list of parents list(list(int)), with the node 0 
    as the root, where u in Pa[v] represents that v i the parent of u.
    
    '''
    
    Pa= [list() for i in range(n)]
    queue= [0]
    added= set()
    while len(queue):
        pa= queue.pop()
        added.add(pa)
        for ch in adj[pa]:
            if ch not in added:
                Pa[pa].append(ch)
                queue.append(ch)
    
    return pa

def fromEdgesToParents(n,E):
    '''
    Given a list of edges List((u,v)) representing a decomposable graph obtains 
    the directed representation in terms of a list of parents list(list(int)), 
    with the node 0 as the root, where u in Pa[v] represents that v i the parent 
    of u.
    '''
    
    return fromAdjacencyListToParents(n, fromEdgesToAdjacencyList(n, E))

def getEdgeListDiference(n,E,D):
    '''
    Obtains the difference of the list of edges E minus the list of edges D, E / D
    '''
    
    diff= list()
    adjE= fromEdgesToAdjacencyList(n, E)
    adjD= fromEdgesToAdjacencyList(n, D)
    
    for u in range(n):
        for v in adjE[u]:
            if v>u:
                if v not in adjD[u]:
                    diff.append((u,v))

    return diff

def joinEdges(n,E,F):
    adjE= fromEdgesToAdjacencyList(n, E)
    adjF= fromEdgesToAdjacencyList(n, F)
    
    for u in range(n):
        for v in adjF[u]:
            if v not in adjE[u]:
                adjE[u].append(v)
    
    return fromAdjacencyListToEdges(adjE)

def areEqualListOfEdges(n,E,F):
    
    adjE= fromEdgesToAdjacencyList(n,E)
    adjF= fromEdgesToAdjacencyList(n,F)
    
    for i in range(n):
        set1= set(adjE[i])
        set2= set(adjF[i])
        
        if len(set1.symmetric_difference(set2))>0:
            return False
    
    return True

def getAllPathsFromVertex(adj,X,maxDist=np.Inf):
    '''
    Given a forest obtains all the (connected maximal) paths starting from the given vertex
    adj: adjacency list
    X: index of a vertex
    '''
    paths= list()
    for ch in adj[X]:
        __indepth([X,ch],adj,paths,maxDist)
    
    return paths
        
def __indepth(part,adj,paths, maxDist):
    '''
    Auxiliary recursive method that produces all the (connected maximal) paths given a partial path
    of length higher than 2, and adjacency list and a list of paths that is actualized 
    '''
    Ch= adj[part[-1]]
    if len(Ch)==1 or len(part)>= maxDist:
        #If it is a leaf or its too far
        paths.append(part[:])
    else:
        #If it is an internal node and is not too far
        for ch in Ch:
            if ch!= part[-2]:
                part.append(ch)
                __indepth(part,adj,paths,maxDist)
                del part[-1]



def entropy(D,S, r= None):
    '''
    D: data set, np.array(num.instances x num.features)
    S: subset of features, list[int]
    r: the cardinality of the subset of features, list[int]

    return the empirical entropy of the multivariate variable X_S
    '''
    d= len(S)
    n= D.shape[0]
    if r is None:
        r= np.unique(D[:, S], axis=1)
    x= S[:,0]
    w= 1
    for i in range(1,d):
        w*= r[i-1]
        x+= D[:,S[i]]*  w
    p= np.unique(x,return_counts=True)[1]/n
    return np.sum(-p*np.log2(p))

def pairwiseMutInf(D,return_entropy= False):
    '''
    Compute the empirical pairwise mutual information between each pair of variables
    D: data set, np.numpy(num.instances x num.variables)
    return_entropy: False-> I[i,i]=-inf, True->I[i,i]=H[i]
    '''

    n,d= D.shape
    #H(Xi)
    H= np.zeros(d)
    # I(Xi,Xj)
    I= np.zeros((d,d))

    r= np.zeros(d)
    for i in range(d):
        (x,p)= np.unique(D[:,-i],return_counts=True)
        r[i]= len(x)
        p=p/n
        H[i]= np.sum(-p[i]* np.log2(p[i]) for i in range(len(p)))

        if return_entropy:
            I[i, i]= H[i]
        else:
            I[i,i]= -np.inf

    for i in range(d):
        for j in range(i+1,d):
            (x, p) = np.unique(D[:, i]*r[j] + D[:,j], return_counts=True)
            p= p/n
            I[i,j] = H[i]+H[j]-np.sum(-p[i] * np.log2(p[i]) for i in range(len(p)))
            I[j,i] = I[i,j]

    return I

def maximumWeigtedTree(W):
    '''
    A symmetric weight matrix W np.array(n x n)

    The method implements Prim's algorithm for finding a maximum weighted tree

    Inefficient implementation: O(n^3) -- related to tetrahedral numbers, n(n+1)*(n+2)/6)
    By using the sort operator: O(n^2 log n)

    Return the maximum weighted tree over the n indices. The tree is directed with 0 index
    as the root.
    '''
    (n,n)= W.shape

    remain = [i for i in range(1,n)]
    added = [0]
    Pa= [list() for i in range(n)]
    added_weight= 0
    while (len(added) < n):
        maximum = -np.inf
        a = -1
        b = -1
        for i in added:
            for j in remain:
                if maximum < W[i,j]:
                    maximum = W[i,j]
                    a = i
                    b = j

        added_weight+= maximum
        Pa[b].append(a)
        added.append(b)
        remain.remove(b)

    return Pa,added_weight

def maxLikelihoodTree(D,S= None):
    '''
    Learn a tree that maximizes the likelihood

    D: data set, np.array(num.inst x num.vars).
    S: the subset of variables. If None the tree is learnt using all the variables

    return a directed version of the maximum likelihood tree where the root is the first var
    '''

    if S is not None:
        D= D[:,S]

    I= pairwiseMutInf(D)
    return maximumWeigtedTree(I)

def randomTree(n,seed= 0):
    '''
    Return a (directed) random tree
    '''
    W= np.zeros((n,n))
    for i in range(n):
        W[i,i+1:]= np.random.random(n-1-i)
        W[i+1:,i]= W[i,i+1:]

    return maximumWeigtedTree(W)[0]

def plot_graph_from_adjacency_list(Pa, title, file_name='graph.png', show_nets=True):

    n= len(Pa)
    # Create a graph from the adjacency list
    G = nx.Graph()
    # Add n nodes to the graph
    G.add_nodes_from(range(n))
    for node, neighbors in enumerate(Pa):
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Calculate node sizes
    degrees = dict(G.degree())
    num_nodes = len(Pa)
    max_diameter = 5000 / num_nodes  # Maximum size proportional to the number of nodes
    node_sizes = [max_diameter * degrees[node] for node in G.nodes()]

    # colors
    degree_values = np.array(list(degrees.values()))
    max_degree = max(degree_values)
    min_degree = min(degree_values)

    # Normalize degree values to [0, 1] range for colormap
    norm = plt.Normalize(vmin=min_degree, vmax=max_degree)

    # Choose a colormap
    colormap = plt.get_cmap('coolwarm')  # From blue to red

    # Map node degrees to colors
    node_colors = [colormap(norm(degrees[node])) for node in G.nodes()]

    # Create the plot
    pos = nx.kamada_kawai_layout(G)  # Layout for a visually appealing arrangement

    # Create the figure and axis
    plt.figure(figsize=(12, 12))
    plt.title(title)

    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black')

    # Remove axis for cleaner visualization
    plt.axis('off')

    # Get graph statistics
    stats = network_statistics(G)

    # Create a legend for statistics
    legend_elements = [
        plt.Line2D([0], [0], marker='', color='w', label=f'Average Degree: {stats["Average Degree"]:.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Diameter: {stats["Diameter"]}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Average Path Length: {stats["Average Path Length"]:.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Number of Nodes: {stats["Number of Nodes"]}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Number of Edges: {stats["Number of Edges"]}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Density: {stats["Density"]:.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w',
                   label=f'Average Clustering Coefficient: {stats["Average Clustering Coefficient"]:.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Degree Assortativity: {stats["Degree Assortativity"]:.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Avg. Degree Centr.: {np.average(list(stats["Degree Centrality"].values())):.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Avg. Betweenness Centr.: {np.average(list(stats["Betweenness Centrality"].values())):.2f}',
                   markersize=10, markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='', color='w', label=f'Avg. Closeness Centr.: {np.average(list(stats["Closeness Centrality"].values())):.2f}',
                   markersize=10, markerfacecolor='gray')
    ]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)

    # Add a title
    plt.title(title, fontsize=16)
    if show_nets:
        plt.show()

    plt.savefig(f'./results/{file_name}')
    plt.close()
