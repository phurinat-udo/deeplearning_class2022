# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
# %%
path = ''
all_files = glob.glob(os.path.join(path, "*.csv"))
df = {}
for f in all_files:
    print(f)
    df[f] = pd.read_csv(str(f))
# %%
demogra_dict = pd.read_excel('NAHNES 2014 Dictionary.xlsx')
demogra_dict
# %%
alldf = df['diet.csv'].merge(df['demographic.csv'], how='left', left_on='SEQN', right_on='SEQN')
alldf = alldf.merge(df['labs.csv'], how='left', left_on='SEQN', right_on='SEQN')
# alldf = alldf.merge(df['examination.csv'], how='left', left_on='SEQN', right_on='SEQN')
# alldf = alldf.merge(df['medications_enc.csv'], how='left', left_on='SEQN', right_on='SEQN')
# alldf = alldf.merge(df['questionnaire.csv'], how='left', left_on='SEQN', right_on='SEQN')
alldf
# %%
n = 0
for i in alldf.columns:
    if alldf[i].isna().sum() > 5000:
        alldf.drop(i, axis=1, inplace=True)
    # print(alldf[i].isna().sum())
# %%
all_dict = {}
for i in demogra_dict.index:
    if demogra_dict.loc[i, 'Variable Name'] in alldf.columns:
        print(demogra_dict.iloc[i])
# %%
corr_matrix=alldf.corr()
# %%
fig, ax = plt.subplots(figsize=(12,10))         # Sample figsize in inches
ax.matshow(corr_matrix)
plt.show()
# %%
corr_matrix.style.background_gradient(cmap='coolwarm')

# %%
