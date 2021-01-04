# -*- coding: utf-8 -*-
"""
Created on 2020/11/12  17:13 

# Author： Jinyu
"""
import  matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import *

os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))

datafile = 'iteration_r.csv'
data = pd.read_csv(datafile)
demand_data=pd.read_csv('keji_final.csv')
ot_fre={('ANC', 'SEA'):2, ('ANC', 'LAX'):1, ('ANC', 'PDX'): 1, ('LAX', 'PDX'):4, ('LAX', 'SEA'):7, ('LAX', 'ANC'): 1, ('PDX', 'LAX'): 4, ('PDX', 'SEA'): 3,
 ('PDX', 'ANC'):1, ('SEA', 'LAX'):3, ('SEA', 'PDX'): 7, ('SEA', 'ANC'): 2}
ot_f=ot_fre.values()

segment_info=[(dept, arrive,demand,fre,ttime)for (dept, arrive,demand,fre,ttime) in zip(demand_data['Dept'].tolist(), demand_data['Arrive'].tolist(),
            demand_data['Demand'].tolist(),demand_data['frequency'].tolist(),demand_data['travel_time'].tolist())]
market = ['M' + str((s[0], s[1])) for s in segment_info]
demand=[info[2]/100 for info in segment_info]
fre=[info[3] for info in segment_info]
ttime=[info[4]/60 for info in segment_info]
width = 0.05
#market_list=['M1','M2','M3']
#passenger_type_list=['oneway', 'outbound', 'inbound']
player_list=['AS','OT']
x=np.arange(len(market))
# data_NW=data.loc[data['player']=='AS']
# data_UA=data.loc[data['player']=='others']
width = 0.3
fig=plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xticks(x +width/2)#将坐标设置在指定位置
ax.set_xticklabels(x)#将横坐标替换成

b1=ax.bar(x,demand,width,alpha = 0.9,label='Demand/100')
b2=ax.bar(x+width,ttime,width,alpha = 0.9,color= 'red',tick_label = market,label='Frequency_AS')
b3=ax.bar(x+2*width,fre,width,alpha = 0.9,color= 'g',label='Frequency_AS')
b4 = ax.bar(x+2*width, ot_f, width, bottom=fre,color='y',label='Frequency_OT')

ax.set_xlabel('Market')

for a,b in zip(x,demand):   #柱子上的数字显示
    plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=8) #a位置，b数值，然后是京都
for a,b,c in zip(x+2*width,fre,ot_f):
    plt.text(a,b,'%.f'%b,ha='center',va='bottom',fontsize=8)
    plt.text(a,b+c, '%.f'% c, ha='center', va='bottom', fontsize=8)
for a,b in zip(x+width,ttime):
    plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=8)
plt.legend(handles = [b1,b2,b3,b4], labels = ['Demand/100', 'Travel Time/hour','Frequency_AS','Frequency_OT'],loc=0)
plt.title('Market info',fontsize=10)
plt.show()
