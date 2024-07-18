import pandas as pd
import numpy as np

path = "/home/HTNclinical/Select-Optimal-Decisions-via-DRO-KNN-master/training-data/HTN_RegistryOutput.csv"
df = pd.read_csv(path,nrows=1000,on_bad_lines='skip')
print(df.shape)
print(df.columns)