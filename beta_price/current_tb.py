# -*- coding: utf-8 -*-
"""
Created on 2020/12/12  15:51 

# Author： Jinyu
"""
import  matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

path = 'F:\jinyu visit\Dropbox\Pyomo_in_China\Pyomo3_smallPDX\data'
os.chdir(path)

Aggregate_Predicted=[-0.448828711,-0.14182689,0.121650192,0.310974078,0.414068152,0.44109491,0.419505986,0.382157701,0.354671726,0.349022273
,0.365539848,0.398641456,0.439102449,0.470527292,0.465569115,0.390300277,0.218939745,-0.0492247,-0.383459532,-0.731996008]
a=np.array(Aggregate_Predicted)+1
time=list(range(4,24))


df=pd.read_csv('AS_current_timetable_70.csv')

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
airport=['ANC','SEA','LAX','PDX']
maker=['o', '^', '*','x',]
color=['b', 'r', 'g','k']
marker_AP=dict(zip(airport,maker))
color_AP=dict(zip(airport,color))

for i, ap1 in enumerate(airport):
    scatter_record={}
    for j ,ap2 in enumerate(airport):
        if ap1 !=ap2:
            df_small=df[(df['ORIGIN'] == ap1)&(df['DEST'] == ap2)]
            d_time=(df_small['CRS_DEP_TIME']/60).tolist()
            arr=(df_small['CRS_ARR_TIME']/60).tolist()
            arr_time=[25 if x <6 else x for x in arr]
            a=ax1.scatter([ap1]*len(d_time), [d_time], linewidth=1, color=color_AP[ap1], marker=marker_AP[ap2],label='from '+ap1+' to '+ap2)
            for t in range(len(df_small)):
                #ax1.scatter([ap1],[d_time[t]],linewidth=1,color=color_AP[ap1],marker=marker_AP[ap2],label='dest airport '+ap2)
                ax1.annotate("",xy=(ap2, arr_time[t]),xytext=(ap1,d_time[t]),arrowprops=dict(arrowstyle="->", color=color_AP[ap1]))
ax1.plot(Aggregate_Predicted,time)
ax1.set_ylim(0,5)
ax1.set_ylim(5,25)
ax1.set_yticks(np.linspace(6,24,19))
ax1.set_yticklabels([str(i)+':00' for i in range(6,25,1)])
#ax1.arrow[]
for ap in airport:
    ax1.axvline(x=ap,color=color_AP[ap])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel('Time')
ax1.set_xlabel('Airport')
ax1.set_title('AS Current Timetable',)
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.12),ncol=6)

plt.show()
