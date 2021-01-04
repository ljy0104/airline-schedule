# -*- coding: utf-8 -*-
"""
Created on 2020/12/8  15:18 

# Author： Jinyu
"""
import  matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import xlrd
import dateutil
from matplotlib.lines import Line2D
from matplotlib.font_manager import *

path = 'F:\jinyu visit\Dropbox\Pyomo_in_China\Pyomo3_smallPDX\data'
os.chdir(path)

def timetable_plot(path,excel_name,title):
    Aggregate_Predicted = [-0.448828711, -0.14182689, 0.121650192, 0.310974078, 0.414068152, 0.44109491, 0.419505986,
                           0.382157701, 0.354671726, 0.349022273
        , 0.365539848, 0.398641456, 0.439102449, 0.470527292, 0.465569115, 0.390300277, 0.218939745, -0.0492247,
                           -0.383459532, -0.731996008]
    a = np.array(Aggregate_Predicted) + 1
    time = list(range(4, 24))
    dir = path
    os.chdir(dir)
    wb = xlrd.open_workbook(excel_name)
    sheets = wb.sheet_names()#retun a list with all sheet names
    fig = plt.figure(figsize=(12,12))
    for i,name in enumerate(sheets):
        df=pd.read_excel(excel_name,sheet_name=name)
        ax1 = fig.add_subplot(2,5, i+1)
        ax1.set_title('iteration' + str(i + 1))
        airport = ['ANC', 'SEA', 'LAX', 'PDX']
        maker = ['o', '^', '*', 'x', ]
        color = ['b', 'r', 'g', 'k']
        marker_AP = dict(zip(airport, maker))
        color_AP = dict(zip(airport, color))
        for i, ap1 in enumerate(airport):
            for j, ap2 in enumerate(airport):
                if ap1 != ap2:
                    df_small = df[(df['ORIGIN'] == ap1) & (df['DEST'] == ap2)]
                    d_time = (df_small['CRS_DEP_TIME'] / 60).tolist()
                    arr = (df_small['CRS_ARR_TIME'] / 60).tolist()
                    arr_time = [25 if x < 6 else x for x in arr]
                    for t in range(len(df_small)):
                        ax1.scatter([ap1], [d_time[t]], linewidth=1, color=color_AP[ap1], marker=marker_AP[ap2])
                        ax1.annotate("", xy=(ap2, arr_time[t]), xytext=(ap1, d_time[t]),
                                     arrowprops=dict(arrowstyle="->", color=color_AP[ap1]))
        ax1.plot(Aggregate_Predicted,time,linestyle='--',color='darkviolet')
        ax1.plot(a, time,linestyle='--',color='darkviolet')
        ax1.plot(a+1, time,linestyle='--',color='darkviolet')
        ax1.plot(a+2, time,linestyle='--',color='darkviolet')
        ax1.set_ylim(0, 5)
        ax1.set_ylim(5, 25)
        ax1.set_yticks(np.linspace(6, 24, 19))
        ax1.set_yticklabels([str(i) + ':00' for i in range(6, 25, 1)])
        # ax1.arrow[]
        for ap in airport:
            ax1.axvline(x=ap, color=color_AP[ap])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylabel('Time')
        ax1.set_xlabel('Airport')

        #ax1.set_title('AS Current Timetable', )
    plt.show()
        #plt.xticks(range(4),['ANC','SEA','LAX','PDX'])

if __name__ == "__main__":
    path='F:\jinyu visit\Dropbox\Pyomo_in_China\Pyomo3_smallPDX\总结结果\\5小时60航班时差调'
    os.chdir(path)
    timetable_plot(path,'timetable_compile.xlsx','title')

