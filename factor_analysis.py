# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:01:39 2023

@author: Takada Satoshi
"""

from sklearn.decomposition import FactorAnalysis as FA
import pandas as pd

df = pd.read_csv("Score_Data.csv")
df.head()

#不要なDataID列削除
df = df.drop('DataID', axis=1)
df.head()

# 以下のプログラムの参考元：https://note.com/jukinonote/n/n50a644a15981
n_factors = 3
fa = FA(n_factors)
fa.fit(df)

# 因子負荷量を求める
df_factor_loading = pd.DataFrame(fa.components_.T, columns=["factor{}".format(num) for num in range(n_factors)],
                               index=df.columns)
df_factor_loading #因子負荷量 

#因子得点を求める
factor_score = fa.transform(df)
factor_scores = pd.DataFrame(factor_score, columns=["factor_{}".format(num) for num in range(n_factors)],
                           index=df.index)
factor_scores #因子得点 