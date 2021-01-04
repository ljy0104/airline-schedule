# -*- coding: utf-8 -*-
"""
Created on 2020/12/16  16:31 

# Author： Jinyu
"""
import  matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
Aggregate_Predicted=[-0.448828711,-0.14182689,0.121650192,0.310974078,0.414068152,0.44109491,0.419505986,0.382157701,0.354671726,0.349022273
,0.365539848,0.398641456,0.439102449,0.470527292,0.465569115,0.390300277,0.218939745,-0.0492247,-0.383459532,-0.731996008]
a=np.array(Aggregate_Predicted)+1
time=list(range(4,24))

proportions = [0.248772702, 0.124024325, 0.171029853, 0.10838055, 0.213028019, 0.134764552]
# proportions.reverse()
# type_name = ['a', 'b', 'c', 'd', 'e', 'f']
# pax_input=(type_name,proportions)
# peak_time = [5, 8, 11, 14, 17, 20]
# passenger_peak_time = dict(zip(type_name, peak_time))
# passenger_proportions = dict(zip(type_name, proportions))
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(111)

ax1.plot(Aggregate_Predicted,time,)
ax1.plot(a,time)
# ax1.set_ylim(0, 5)
# ax1.set_ylim(5, 25)
# ax1.set_yticks(np.linspace(6, 24, 19))
# ax1.set_yticklabels([str(i) + ':00' for i in range(6, 25, 1)])
        # ax1.arrow[]
plt.show()