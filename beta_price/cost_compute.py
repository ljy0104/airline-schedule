# -*- coding: utf-8 -*-
"""
Created on 2020/11/17  15:12 

# Author： Jinyu
"""
import os
import pandas as pd
from data_prepare import operation_cost
os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))

df_keji=pd.read_csv('keji_final.csv')
df_AS=pd.read_csv('AS_current_timetable_70.csv')
df_others=pd.read_csv('other_current_timetable35.csv')

market_distance={(ap1,ap2):distance for (ap1,ap2,distance) in zip(df_keji['Dept'].tolist(),df_keji['Arrive'].tolist(),df_keji['Nonstop_Miles'].tolist())}
tuple_list_AS=[(market_distance[(ap1,ap2)],capacity)for (ap1,ap2,capacity) in zip(df_AS['ORIGIN'].tolist(),df_AS['DEST'].tolist(),df_AS['Capacity'].tolist())]
tuple_list_OT=[(market_distance[(ap1,ap2)],capacity)for (ap1,ap2,capacity) in zip(df_others['ORIGIN'].tolist(),df_others['DEST'].tolist(),df_others['Capacity'].tolist())]
# a=[(market_distance[(ap1,ap2],capacity) for ]
c_AS=operation_cost(tuple_list_AS)
c_OT=operation_cost(tuple_list_OT)

print(c_AS)
print(c_OT)