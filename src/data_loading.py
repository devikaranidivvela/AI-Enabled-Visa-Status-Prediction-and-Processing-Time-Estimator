#importing the Dependencies
import warnings
warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Library to split data
from sklearn.model_selection import train_test_split

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
def load_data():
    df=pd.read_csv("C:\Users\DEVIKA\Desktop\VISA-PROCESSING\data\us_perm_visas.csv")
    return df
