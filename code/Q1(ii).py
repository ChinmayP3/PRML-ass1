import numpy as np
import matplotlib.pyplot as plt



def display(p1,p2):
        plt.scatter(p1, p2, color='g', marker='*')
        plt.title('plotting after PCA with centering')
        plt.xlabel('principle component 1')
        plt.ylabel('principle component 2')
        plt.show()

data = np.genfromtxt("Dataset.csv", delimiter=',')


meanData = data.mean(0)
print("Mean :")

 

data = data.T

size = data.shape[1]

CovarianeMatrix = data@(data.transpose())/size

print("Mean of the Data:", data.mean(axis=1))

EigenValue, EigenVector = np.linalg.eig(CovarianeMatrix)

index = EigenValue.argsort()[:: -1]
EigenValue = EigenValue[index]
EigenVector = EigenVector[:, index]
print(len(EigenVector))

s = ((EigenVector.T)@data)


PrincipleComp = EigenVector
principalComponent1 = PrincipleComp[:, 0:1]
principalComponent2 = PrincipleComp[:, 1:2]


xTCx1=(principalComponent1.T)@CovarianeMatrix@principalComponent1
xTCx2=(principalComponent2.T)@CovarianeMatrix@principalComponent2
variance1 = (xTCx1)[0][0]
variance2 = (xTCx2)[0][0]
print("Principal Component 1:", principalComponent1)
print("Principal Component 2:", principalComponent2)
data2 = data.transpose()
weight = data2@principalComponent1
i = 0
while(i<len(weight)):
        plt.scatter(principalComponent1[0]*weight[i],principalComponent1[1]*weight[i],c='red')
        i+=1
i=0
while(i<len(weight)):
        plt.scatter(principalComponent2[0]*weight[i],principalComponent2[1]*weight[i],c='blue')
        i+=1
plt.show()

print()
print("Variance by pc1:", variance1)
print("Variance by pc1:", variance2)
pc1 = variance1/(variance1+variance2)
pc2 = variance2/(variance1+variance2)
print("% of variance explained by principal component 1: ",
        (variance1/(variance1+variance2)*100))
print("% of variance explained by principal component 2: ",
        (variance2/(variance1+variance2) * 100))

# Ploting the graph  
display(s[0],s[1]) 
 
