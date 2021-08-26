from datetime import datetime
from math import isnan
from tqdm import tqdm
from helper import histograms

import numpy as np
import pandas as pd



outputname = '/home/chrei/code/quantifiedSelfData/netatmo_daily_summaries.csv'
df = pd.read_csv('/home/chrei/PycharmProjects/correlate/0_data_raw/MFP/meals.csv')  # , index_col=0

sugar = []
Cholest = []
