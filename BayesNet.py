import numpy as np

#np.ones: vector de 1-s de longitud 10 y de tipo entero.
#Pa: set of parents
#Orden ancestral de las variables: cada nodo tiene una posicion anterior a cualquier descendiente suyo

class BayesNet:
    #Si inicializas un parametro en init lo puedes usar en toda la clase
    def __init__(self,n,r):
        '''
        Constructor
        n: number of (discrete) random variables
        r: cardinality of the (discrete) random variables, np.array(int)
        and: ancestral order of the variables
        Pa: the list of parents of each variable, list(list(int))
        D: training data
        w: weights associated to the instances in the training data
        minK,maxK: minimum and maximum (if possible) number of parents for each variable
        alpha: hyperparameter of the Dirichlet distribution. When D!=None alpha is used in the Bayesian estimate of the parameters
            when D==None the parameters are randomly sampled from a Disrichlet distribution with alpha=alpha_1,..., alpha_rx parameters 
            for each value of Pa
        seed: random seed
        '''
        #Number of variables
        self.n=n
        #Cardinality of each variable
        #We initialize it as if every variable was binary: r=2
        self.r=r
        self.Pa= list(np.array([]) for i in range(n))

    def copy(self):
        '''
        Returns a copy of the model
        '''
        #Copias de los arrays
        model=BayesNet(self.n,self.r.copy())
        #Ancestral order of the variables
        model.anc=[i for i in self.anc]
        #Parents of each variable
        model.Pa= [[i for i in Pa] for Pa in self.Pa]
        model.cpt=[cpt.copy() for cpt in self.cpt]
        return model
        
    def learn(self,anc,Pa,D,w=None,alpha=0):
        '''
        
        '''
        #Ancestral order of the variables
        self.anc=anc
        #Parents of each variable
        self.Pa= Pa
        #Condicional probability tables in ancestral order
        #probabilidades condicionales de cada variable condicionada a cada uno de sus padres
        #r[i] cardinality of the variables, r[j] cardinality of the parents
        self.cpt=[CPT(X=i,Pa=Pa[i],rx=self.r[i],rpa=[self.r[j] for j in Pa[i]],D=D,w=w,alpha=alpha) for i in range(self.n)]
    
    
    def paramUnifPriors(self,N,sampSize=1):
        '''
        Tomamos paramUnifPriors de la clase CPT
        '''
        for i in range(self.n):
            self.cpt[i].paramUnifPriors(N,sampSize)
    
    
    def random(self,minK=2,maxK=3,alpha=1,seed=None):
        '''
        Randomly generates a decomposable model (structure and parameters) with a bounded clique size
        maxK: maximum clique size
        minK: minimum clique size
        For trees: maxK=minK=2
        alpha: parameter of the Dirichlet distribution used for sampling the parameters
        seed: random seed used in numpy.random.seed()
        
        clique: subset of vertices of an undirected graph such that its induced subgraph is complete:
        every two distinct vertices in the clique are adjacent.
        '''
        
        if seed!=None: 
            np.random.seed(seed)
        
        remain= range(self.n)
        #random.shuffle: randomize the elements of the array in question
        np.random.shuffle(remain)
        
        #random ancestral order
        #np.fliplr: flips the columns of the array (como si fuese un espejo)
        #np.ndarray.tolist: convierte el array en lista
        #randomize the ancestral order of variables
        
        self.anc=np.ndarray.tolist(np.fliplr([remain])[0])

        cliques=[]
        self.Pa=[[] for i in range(self.n)]
        #Create the first clique
        #.pop(): removes and returns the last element in the list
        #np.random.choice: generates a random sample from an array
        #??????? (maxK+1-minK)+minK
        
        C=[remain.pop() for i in range(np.random.choice(maxK+1-minK)+minK)]
        for i in range(1,len(C)):
            self.Pa[C[i]]=[C[j] for j in range(i)]
        cliques.append(C)
        
        #Create the remaining cliques
        while remain:
            X= remain.pop()
            ancC= cliques[np.random.choice(len(cliques))]
            #take a number of ancestors uniformly at random in the interval [min(numAnces,minK),...,min(numAnces,maxK)] 
            self.Pa[X]= np.ndarray.tolist(np.random.choice(ancC,np.min((np.random.choice(maxK-minK+1)+minK,len(ancC))), replace=False))
            C=[X]
            C.extend(self.Pa[X])
            cliques.append(C)
        
        self.cpt=[CPT(i,self.Pa[i],self.r[i],[self.r[j] for j in self.Pa[i]],alpha) for i in range(self.n)]
        [cpt.random(alpha,seed) for cpt in self.cpt]
        
            
    def eval(self,x):
        '''
        Evaluate an instance
        '''
        #producto de todos los elementos del vector de cpt
        return np.prod([self.cpt[i].getProb(x) for i in self.anc])
        
    
    '''
    ALGORITMO DE CLASIFICACION
    x: instancia que queremos clasificar
    ind: indice de la variable clase
    '''
    def argMax(self,x,ind):
        '''
        The value of the variable with maximum probability
        x: instancia que queremos clasificar
        ind: indice de la variable clase
        '''
        copy= x.copy()
        #Valor inicial: C=0
        copy[ind]=0
        argmax= 0
        #evaluate the instance's probability
        val=self.eval(copy)
        max= val
        for arg in range(1,self.r[ind]):
            copy[ind]=arg
            val=self.eval(copy)
            if(val>max):
                argmax=arg
                max=val
            
        return argmax
        

    #log likelihood
    def LL(self,D):
        '''
        Log likelihood of the data set given the model
        '''
        return np.sum(np.log2([self.eval(x) for x in D]))

    def sample(self,m=1,seed=None):
        '''
        Random sampling of N instances according to the probability represented by
        the decomposable model
        '''
        if seed!=None: np.random.seed(seed)
        
        if m==1:
            X= np.zeros(self.n,dtype=np.int)
            for j in self.anc:
                X[j]= self.cpt[j].sample(X)
        else:
            X=np.zeros((m,self.n),dtype=np.int)
            for i in range(m):
                for j in self.anc:
                    X[i,j]= self.cpt[j].sample(X[i])
        
        return X

#Conditional probability tables

class CPT:
    def __init__(self,X,Pa,rx,rpa,D=None,w=None,seed=1,alpha=None):
        #Index of the variable
        self.X=X
        #Indices of the conditioning variables
        self.Pa=Pa
        #Number of conditioning variables
        self.m=len(Pa)
        #cardinality of the variable
        self.rx=rx
        #cardinality of the parents
        self.rpa= rpa
        #weight of the conditioning variables for computing the conditioning index
        self.wpa= np.ones(len(rpa), int)
        #for i in range tambien incluye el 0
        for i in range(len(rpa)-1):
            self.wpa[i+1]=self.wpa[i] * rpa[i]
        
        if not (D is None):
            if len(np.shape(D))>0:
                self.learn(D=D,w=w,alpha=alpha)
            else:
                self.random(D=D,seed=seed)
    
    def copy(self):
        cpt=CPT(self.X,self.Pa,self.rx,self.rpa)
        cpt.cpt=np.copy(self.cpt)
        return cpt
    
    def learn(self,D,w=None,alpha=0):
        #n number of instances
        n=D.shape[0]

        #Equally weighted cases
        if w==None: w= np.ones(n, dtype=np.float)
        
        #length of the set of parents
        if(self.m>0):#conditional probability
            #creamos un vector de ceros con cada componente de longitud igual al producto entre el cardinal de la
            #variable y el cardinal de los padres
            
            self.cpt= np.ones((np.prod(self.rpa),self.rx),dtype=np.float)*float(alpha)/self.rx
            for i in range(n):
                #???????????
                x= D[i]
                #Le sumamos el peso de cada variable
                #Pa[j] son los indices de las variables condicionantes. x[Pa[j]] es el valor de esa variable.
                #Lo multiplicamos por su peso.
                self.cpt[np.sum([x[self.Pa[j]]*self.wpa[j] for j in range(len(self.Pa))])][x[self.X]]+= w[i]
            #sum in the second axis, x
            #Se computa la probabilidad condicionada para cada variable
            sum= np.sum(self.cpt,1)
            #where returns the indexes where condition is true
            zeros= np.where(sum==0)
            sum[zeros]= 1
            #probabilidad igual para cada uno
            self.cpt[zeros]= np.ones(self.rx)*1.0/self.rx
            self.cpt= (self.cpt.transpose()/sum).transpose()
        
        #if there isn't any set of parents
        else:#marginal probability
            self.cpt= np.ones(self.rx,dtype=np.float)*float(alpha)/self.rx
            for i in range(n):
                #Chapuza: necesito indicar que es int para evitar problemas
                self.cpt[int(D[i][self.X])]+= w[i]
            sum= np.sum(self.cpt)
            
            if not sum==0:
                #Para computar la probabilidad condicional se dividen los pesos entre la suma total.
                self.cpt= self.cpt/sum
            else:
                self.cpt= np.ones(self.rx)/self.rx
    
    def paramUnifPriors(self,N,sampSize=1):
        if len(self.cpt.shape)==1:
            rx= self.cpt.shape[0]
            self.cpt= (self.cpt*N + np.ones(rx)*(sampSize/np.float(rx)))/(sampSize+N)
        else:
            (rpa,rx)=self.cpt.shape
            self.cpt= (self.cpt*N + np.ones((rpa,rx))*(sampSize/np.float(rpa*rx)))/(sampSize/np.float(rpa)+N)

    def random(self,alpha,seed=None):
        if seed!=None: np.random.seed(seed)

        if(self.m>0):#conditional probability
            self.cpt= np.random.dirichlet(np.ones(self.rx)*alpha,np.prod(self.rpa))
        else:#marginal probability
            self.cpt= np.random.dirichlet(np.ones(self.rx)*alpha,1)[0]
    
    def getCond(self,x):
        if(self.m>0):
            return self.cpt[np.sum([x[self.Pa[i]]*self.wpa[i] for i in range(self.m)])]
        else:
            return self.cpt
    
    def getProb(self,x):
        #X index of the variables, Pa index of the conditioning variables
        if(self.m>0):
            #probabilidad condicionada de x a cada valor de Pa
            return self.cpt[np.sum([x[self.Pa[i]]*self.wpa[i] for i in range(self.m)])][x[self.X]]
        else:
            #Probabilidad de x
            return self.cpt[x[self.X]]
            
    def sample(self,x):
        return np.random.choice(a=self.rx,p=self.getCond(x))
        
            
