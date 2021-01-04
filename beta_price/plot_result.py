# -*- coding: utf-8 -*-
"""
Created on 2020/11/15  16:15 

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
itieration=data['Iteration'].astype(int) +1
gap=data['gap']
keji_profit=data['profit']
cost=data['cost']
player_1_equi=data['player_1_equi']-data['cost']
player_2_equi=data['player_2_equi']-327625


fig=plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Profit and Cost in 10 iterations', fontsize='small')
ax1.set_xticks(range(1,11))
ax1.plot(itieration, keji_profit,color='b',  label='playerAS_profit_stage1',ls='-', marker='o',ms=3,lw=2)
ax1.plot(itieration,cost,marker='^',color='k', label='playerAS cost',linestyle='--',ms=3,lw=2)
ax1.plot(itieration, player_1_equi, color='g', label='playerAS profit_in_stage2',marker='o',ms=3,lw=1)
ax1.plot(itieration,player_2_equi,color='r', label='player2 profit in stage2',marker='o',ms=3,lw=1)
#添加数据标签
for x1, keji_profit,cost,player_1_equi,player_2_equi in zip(itieration,keji_profit,cost,player_1_equi,player_2_equi):
    ax1.text(x1, keji_profit, '%.1f'%keji_profit, ha='center', va='bottom', fontsize=6)
    ax1.text(x1, cost, cost, ha='center', va='bottom', fontsize=6)
    ax1.text(x1, player_1_equi, '%.1f'%player_1_equi, ha='center', va='bottom', fontsize=6)
    ax1.text(x1, player_2_equi, '%.1f'%player_2_equi, ha='center', va='bottom', fontsize=6)

ax1.set_xlabel('Iterations')
ax1.set_ylabel('profit')
# color_legend_elements = [Line2D([0], [0], color='b', ),
#                          Line2D([0], [0], color='r', ),
#                          Line2D([0], [0], color='g')]
# line_legend_elements = [Line2D([0], [0], linestyle='-', color='k'),
#                         Line2D([0], [0], linestyle='-.', color='k'),
#                         Line2D([0], [0], linestyle=':', color='k'),
#                         Line2D([0], [0], linestyle='--', color='k')]
ax1.legend(loc = 'best',borderpad=0.5, labelspacing=0.1,frameon=False)#图例边框的内边距
plt.savefig('fix.jpg', dpi=300)
plt.show()
