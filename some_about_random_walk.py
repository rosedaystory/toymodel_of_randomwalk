
# coding: utf-8

# In[1]:

from sklearn.datasets import make_regression
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
from ipykernel import kernelapp as app


# In[337]:


class Rand_walk:
    
    def __init__(self,n_data, n_length):
        self.n_data = n_data
        self.n_length = n_length
        np.random.seed()
        self.rand = np.insert(np.cumsum(np.random.randn(n_data, n_length),axis=1),0,0,axis=1)
        self.rand_df = pd.DataFrame(self.rand.T)

    
    def draw_rand(self):
        """
        draw the random walk.
        """
        plt.plot(range(self.n_length + 1),self.rand_df)
        plt.show()
        
    def std(self,ensemble=True,*args):
        """
        if ensemble == True , calculate every ith ensemble std
        if ensemble != True, calculate each randomwalk std
        default = True
        """
        if ensemble == True:
            return self.rand_df.std(axis=1)
        else:
            return self.rand_df.std(axis=0)
    def mean(self, ensemble=True):
        """
        if ensemble == True , calculate every ith ensemble mean
        if ensemble != True, calculate each randomwalk mean
        default = True
        """
        if ensemble == True:
            return self.rand_df.mean(axis=1)
        else:
            return self.rand_df.mean(axis=0)        
    
    def cov(self,i,j):
        """
        calculate cov of random walk of ith with jth
        """
        return ((self.rand_df.iloc[i] - self.rand_df.iloc[i].mean())*(self.rand_df.iloc[j] - self.rand_df.iloc[j].mean())).mean()
    def std2(self):
        a = np.random.randn(len(self.rand_df[0]),len(self.rand_df[0]))
        b = self.std()
        for i in range(len(self.rand_df[0])):
            for j in range(len(self.rand_df[0])):
                a[i][j] = b[i]*b[j]
        return a
    
    def cov_all(self):
        df_mean = (self.rand_df.T - self.rand_df.T.mean(axis = 0))
        a = np.random.randn(len(self.rand_df[0]),len(self.rand_df[0]))
        for i in range(len(self.rand_df[0])):
            for j in range(len(self.rand_df[0])):
                a[i][j] = (df_mean[i] * df_mean[j]).mean()
        return a
    
    def lo (self,i,j):
        return self.cov(i,j) / self.std()[i] / self.std()[j]
    
    def lo_all(self):
        return self.cov_all() / self.std2()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:





# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



