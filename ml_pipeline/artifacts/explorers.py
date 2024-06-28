import os
import pandas as pd
from scipy.stats import shapiro, pearsonr
from itertools import product
from pprint import pprint
from . import line
import numpy as np
from numba import njit, jit

class Explorer:
    """"Designed to get quick gloabl stats."""
    def __init__(self, csv_filepath: str='', target: str='', data: pd.DataFrame=pd.DataFrame()):
        self.target = target
        self.path = csv_filepath
        self.shapiro_df = None
        # if self.path !='': 
        #     if os.path.exists(self.path):
        #         self.df = pd.read_csv(self.path)
        #     else:
        #         raise Exception(f'{self.path} doesnt exist.')
        #else:
        self.df = data

    @line
    def global_description(self):
        print(self.df.dtypes)
        print(self.df.describe())

    @line
    def global_check_null(self):
        for col in self.df.columns:
            print(f"Null counts {col} : {self.df[col].isnull().sum()}")

    @line
    def distro_top_category(self):
        # nbre de modalités
        for col in self.df.select_dtypes(include='object').columns:
            print(f"{col} has {self.df[col].nunique()} values")
            print(100 * self.df[col].value_counts() / self.df[col].count())
        # graphes

    @line
    def normality(self):
        # tests de normalité
        shapiro_res = {}
        for col in self.df.select_dtypes(exclude='object').columns:
            shapiro_res.update({col: shapiro(self.df[col])[1]})
            self.shapiro_df = pd.DataFrame.from_dict(shapiro_res, orient='index', columns=['pvalue'])
        print(self.shapiro_df)
        # graphes

    @line
    def correlation(self):
        # tests de correlation
        corr_res = []
        for col1, col2 in list(product(self.df.columns, self.df.columns)):
            try:
                corr, xx = pearsonr(self.df[col1], self.df[col2])
                corr_res.append((col1, col2, corr, xx))
            except Exception as e:
                pass
        corr_res = [x for x in corr_res if x[0] == self.target and x[3] <= .05]
        pprint(corr_res)



class IIE:
    """DESIGNED TO GET QUICK INSIGHTS ABOUT EXPLOITABILITY : ENTROPY - INCLUSION - INTERSECTION RATES"""
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2
        assert set(self.df1.columns) == set(self.df2.columns), f"cols must be the same"
        self.columns = list(self.df1.columns)
        self.df1_np = self._get_array(df1)
        self.df2_np = self._get_array(df2)

    def _get_array(self, df):
        """get column oriented array as np array"""
        return np.array([v for v in df.to_dict(orient='list').values()])
    
    #@jit
    def _get_intersection(d1_np, d2_np, i):
        a = len(set(d1_np[i]).intersection(set(d2_np[i])))
        b = len(set(d1_np[i]).union(set(d2_np[i])))
        op = np.divide(a,b) 
        return [op]

    def _compute_intersections(d1_np, d2_np, cols):
        intersections = {}
        for i,col in enumerate(cols):
            intersections[col] = IIE._get_intersection(d1_np, d2_np, i)
        return intersections
        
    #@jit
    def _compute_inclusions(d1_np, d2_np, cols):
        inclusions = {}
        for i,col in enumerate(cols):
            inclusions[col] = [len(set(d1_np[i]).intersection(set(d2_np[i]))) / len(set(d2_np[i]))]
        return inclusions
        
    @jit
    def _get_entropy(pk, L):
        op = - (pk * np.log(pk) / np.log(L)).sum()
        return op 
    
    def _compute_entropies(d, cols):
        entropies = {}
        for i,col in enumerate(cols):
            pk = d[col].value_counts(normalize=True, dropna = False).values
            entropy = IIE._get_entropy(pk, len(d))
            entropies[col] = [entropy]
        return entropies      
    
    def run(self):
        
        self.intersect = pd.DataFrame(IIE._compute_intersections(self.df1_np, self.df2_np, self.columns)).T.rename(columns={0 : "intersection"})
        self.inclusion = pd.DataFrame(IIE._compute_inclusions(self.df1_np, self.df2_np, self.columns)).T.rename(columns={0 : "inclusion"})
        self.entropy = pd.DataFrame(IIE._compute_entropies(self.df1, self.columns)).T.rename(columns={0 : "entropy"})

        metrics = pd.concat([self.intersect, self.inclusion, self.entropy], axis = 1)
        output = metrics.style.applymap(self.highlight_low_inclusion, subset=pd.IndexSlice[:, ['intersection', 'inclusion']])\
             .applymap(self.highlight_low_entropy, subset=pd.IndexSlice[:, ['entropy']])
        return output

    def highlight_low_inclusion(self, value):
        if value > 0.95:
            return 'background-color: green'     
        #else:
        #    return 'background-color: pink'
        
    def highlight_low_entropy(self, value):
        if value > 0.1:
            return 'background-color: green'     
        #else:
        #    return 'background-color: pink'