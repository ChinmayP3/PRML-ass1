import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

X = np.genfromtxt("Dataset.csv", delimiter=',')


X = X.T

Size = X.shape[1]


def polyKernel(x,y,d):
    funxTy=(1+x.transpose()@y)
    return funxTy**d

def getKernelComponentsPolynomial(X,d):
     
    size = X.shape[1]
    N = np.full((size , size), 1/size)
    KernelMatrix = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            KernelMatrix[i,j] = polyKernel(X[:,i],X[:,j],d)

    KernelMatrixCentered = KernelMatrix - N@KernelMatrix - KernelMatrix@N + N@KernelMatrix@N
    eigenValue,eigenVector = np.linalg.eig(KernelMatrixCentered)

    eigenValue=eigenValue.real 
    eigenVector = eigenVector.real

    sortedEigenValues=np.flip(eigenValue.argsort())
    sortedEigenVector = eigenVector[:,sortedEigenValues]
    
    calcLamda=((np.abs(eigenValue))**0.5)
    sortedEigenVector = sortedEigenVector/calcLamda

    kernelSortEV=sortedEigenVector[:,0:2]
    kernel_comps = KernelMatrixCentered.T@(kernelSortEV);
    kernel_comps=kernel_comps.transpose();
                    

    return kernel_comps

 


def llyodsAlgorithm(X,K,muRandom=None,max_iter=1000):
     
    Size = X.shape[1]
    errorTotal = []
    ZReassigned = np.zeros(Size,dtype=np.uint8)
          
    
    muRandom = X[:,np.random.randint(0,X.shape[1],K)]  
    muList = muRandom.copy()

    for iter in range(max_iter):

      
        error=0
        for i in range(Size):
            XCol=X[:,i]
            mucal=muList[:,ZReassigned[i]]
            errorCurrent = ((XCol-mucal)*(XCol-mucal)).sum()
            error += errorCurrent
      

        errorTotal.append(error)
         
        for i in range(Size):
             
            ZReassigned[i] = np.argmin(((X[:,i:i+1]-muList)**2).sum(axis=0))
 

        k=0
        for k in range(K):
             
            isZero=(ZReassigned==k).sum()
            if isZero!=0:
                sumRe=X*((ZReassigned==k).reshape(1,1000))
                muList[:,k] = sumRe.sum(axis=1)/isZero


        if iter>2:
            if abs(errorTotal[-1]-errorTotal[-2])<1:
                break
     
    return muRandom,muList,ZReassigned,errorTotal




def spectralClustering(X,d,num_clusters=4):
      size = X.shape[1]
      N = np.full((size , size), 1/size)
      KernelMatrix = np.zeros((size,size))
      
      for i in range(size):
          for j in range(size):
              KernelMatrix[i,j] = polyKernel(X[:,i],X[:,j],d)

      KernelMatrixCentered = KernelMatrix - N@KernelMatrix - KernelMatrix@N + N@KernelMatrix@N
      eigenValue,eigenVector = np.linalg.eig(KernelMatrix)

      eigenValue=eigenValue.real 
      eigenVector = eigenVector.real

      sortedEigenValues=np.flip(eigenValue.argsort())
      sortedEigenVector = eigenVector[:,sortedEigenValues]

      HCalculate=sortedEigenVector.copy()
      HCalculate=HCalculate[:,:num_clusters].transpose()

     

      HNorm=np.linalg.norm(HCalculate,axis=0).reshape(-1,size)
      HStarCalculate=HCalculate/HNorm
    
   
      muRandom,muList,ZReassigned,history = llyodsAlgorithm(HStarCalculate,K=4,max_iter=1000)

      return HCalculate,HStarCalculate,ZReassigned,muList,muRandom




    
def displayClustered(X,muRandom,muList,ZReassigned):
  
    plt.subplot(1,3,2)
    plt.xlabel("dimension 1 of the Data")
    plt.ylabel("dimension 2 of the Data")
    plt.scatter(X[0],X[1],c=ZReassigned)
    
    plt.title("Spectral relaxation of K-means using KernelPCA k= 4")
    




def displayResults(X,muRandom,muList,ZReassigned):

    plt.figure(figsize=(21,7))

    displayClustered(X,muRandom,muList,ZReassigned)
     
    plt.show()

 

X_centered = X-X.mean(axis=1).reshape(-1,1)
HCalculate,HStarCalculate,ZReassigned,muList,muRandom = spectralClustering(X,2,4)
 
displayResults(X_centered,HCalculate,HStarCalculate,ZReassigned)
 


 

 