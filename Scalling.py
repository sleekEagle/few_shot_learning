import numpy as np
import pandas as pd

class Scaller():
        def __init__(self,data_df):
            super(Scaller, self).__init__()
            self.data_df=data_df
            #calculate stats
            max_ar=data_df['data'].apply(np.max,args=[1])
            maxs=np.max(np.array([item for item in max_ar]),axis=0)
            self.maxs=np.expand_dims(maxs,axis=1)
            min_ar=data_df['data'].apply(np.min,args=[1])
            mins=np.min(np.array([item for item in min_ar]),axis=0)
            self.mins=np.expand_dims(mins,axis=1)
            
        def get_min_max_scaled_row(self,row):
            max_ar=np.repeat(self.maxs,repeats=row.shape[1],axis=1)
            min_ar=np.repeat(self.mins,repeats=row.shape[1],axis=1)
            normalized=(row-min_ar)/(max_ar-min_ar)
            return normalized
        
        def get_min_max_scalled_df(self):
            return self.data_df['data'].apply(self.get_min_max_scaled_row)
