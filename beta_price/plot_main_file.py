# -*- coding: utf-8 -*-
"""
Created on 2020/11/22  14:08 

# Author： Jinyu
"""
import xlrd
import pandas as pd
from pandas import DataFrame
import  matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

def load_plot(path,excel_name,title):
    dir = path
    os.chdir(dir)
    wb = xlrd.open_workbook(excel_name)
    sheets = wb.sheet_names()#retun a list with all sheet names
    av_load_factor=[]
    for i in sheets:
        df=pd.read_excel(title, sheetname=sheets[i])

        pax=df[df['player'] == 'AS']['pax'].sum()

    for sheets_name in sheets:
        df = pd.read_excel(excel_name,sheets_name)
        a=df.loc[df['player'] == 'AS']['load_factor'].mean()
        b=df.loc[df['player'] == 'others']['load_factor'].mean()
        av_load_factor.append((a,b))
    load_AS=[a[0] for a in av_load_factor]
    load_other=[a[1] for a in av_load_factor]
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.set_title('title', fontsize='small')
    ax.plot(sheets, load_AS, color='b', label='AS', ls='-', marker='o', ms=3, lw=2)
    ax.plot(sheets, load_other, marker='^', color='k', label='Other', linestyle='--', ms=3, lw=2)
    for x1, av_load, in zip(sheets, av_load_factor):
        ax.text(x1, av_load[0], str(av_load[0]), ha='center', va='bottom', fontsize=6)
        ax.text(x1, av_load[1], str(av_load[1]), ha='center', va='bottom', fontsize=6)
    ax.set_ylim(0.5, 1)
    # plt.yticks(range(0,10), fontsize=12, rotation=80) # 针对x轴的标签，指定我们想要设定的范围是(0, 10),
    ax.set_xlabel('Iterations')
    ax.set_ylabel('load_factor')
    ax.legend()
    plt.show()


def price_plot(df,df_price,df_timetable,title,segment_info):
    """

    :param df: 
    :param df_price: 
    :param df_timetable: 
    :param title: 
    :param segment_info: 
    """
    market_av_price = {col: df_price[col].tolist()[0] for col in df_price.columns}
    del market_av_price['Unnamed: 0']
    market_damand = {str((s[0], s[1])): s[2] for s in segment_info}

    player_sum_seats={}
    player_sum_pax={}
    player_av_price = {}
    for name in ['AS', 'others']:
        data = df[df['player'] == name]
        data_seats=df_timetable[df_timetable['UNIQUE_CARRIER']==name]
        market=market_av_price.keys()
        a_price = {}
        pax_market={}
        seats_market={}
        for s in market:
            price = data.loc[data['market'] == s]['price'].to_list()
            pax = data.loc[data['market'] == s]['pax'].to_list()
            av_price = np.average(price, weights=pax)
            pax_sum=sum(pax)
            seats=sum(data_seats.loc[(data_seats['ORIGIN']==eval(s)[0])&(data_seats['DEST'] == eval(s)[1])]['Capacity'])
            a_price[s] = av_price
            pax_market[s]=pax_sum
            seats_market[s]=seats
        player_av_price[name] = a_price
        player_sum_pax[name]=pax_market
        player_sum_seats[name]=seats_market
    ot_fre = {('ANC', 'SEA'): 2, ('ANC', 'LAX'): 1, ('ANC', 'PDX'): 1, ('LAX', 'PDX'): 4, ('LAX', 'SEA'): 7,
              ('LAX', 'ANC'): 1, ('PDX', 'LAX'): 4, ('PDX', 'SEA'): 3,
              ('PDX', 'ANC'): 1, ('SEA', 'LAX'): 3, ('SEA', 'PDX'): 7, ('SEA', 'ANC'): 2}

    x = np.arange(len(market_damand))
    ot_f = ot_fre.values()
    color_AS='r'
    color_OT='g'
    fig = plt.figure(figsize=(12,14))
    width = 0.2
    ax = fig.add_subplot(111)
    plt.xticks(x, market_av_price.keys())

    ax.set_title(title, fontsize='small')
    ax.bar(x-width, market_damand.values(), width, color='b', label='Market demand', hatch='/')
    ax.bar(x, player_sum_pax['AS'].values(), width,color=color_AS,edgecolor="k",alpha=0.6,label='Pax in AS')
    ax.bar(x, player_sum_pax['others'].values(), width,color=color_OT,edgecolor="k",alpha=0.6,label='Pax in others',
            bottom=list(player_sum_pax['AS'].values()))
    ax.bar(x+width , player_sum_seats['AS'].values(), width , alpha=0.4,color=color_AS, edgecolor="k", hatch='//',label='AS total seats')
    ax.bar(x + width , player_sum_seats['others'].values(), width ,alpha=0.4, color=color_OT, edgecolor="k", hatch='//',label='others total seats',
           bottom=list(player_sum_seats['AS'].values()))
    ax.spines['top'].set_color('none')

    ax.spines['top'].set_visible(False)

    plt.legend(loc=2)
    ax.set_xlabel('Market',)
    ax.set_ylabel('demand&supply',rotation='horizontal',loc='top',fontsize=12)
    ax1=ax.twinx()
    ax1.plot(market_av_price.keys(), market_av_price.values(), label='Market av_price', color='b', ls='-', marker='o',
            ms=3, lw=2)
    ax1.plot(player_av_price['AS'].keys(), player_av_price['AS'].values(), label='AS av_price', color=color_AS)
    ax1.plot(player_av_price['others'].keys(), player_av_price['others'].values(), label='others av_price', color=color_OT)
    for a, value1, value2, value3, in zip(market_av_price.keys(), market_av_price.values(),
                                          player_av_price['AS'].values(), player_av_price['others'].values()):
        ax1.text(a, value1, '%.2f' % value1, ha='center', va='bottom', fontsize=6)
        ax1.text(a, value2, '%.2f' % value2, ha='center', va='bottom', fontsize=6)
        ax1.text(a, value3, '%.2f' % value3, ha='center', va='bottom', fontsize=6)
    ax1.set_ylabel('price',rotation='horizontal',fontsize=12)
    ax1.spines['top'].set_visible(False)
    # ax = fig.add_subplot(212)
    #
    # ax.bar(x, market_damand.values(),width,color='b', label='Market demand/100',hatch='/')
    # ax.bar(x+ width,fre, width, alpha=0.9, color=color_AS,)
    # ax.bar(x + width, ot_f, width, bottom=fre, color=color_OT, label='Frequency_OT')
    # ax1 = ax.twinx()
    # ax1.bar(market_damand.keys(), market_damand.values(),)
    # for i, value1,value2,value3, in zip(x,market_av_price.values(),player_av_price['AS'].values(),player_av_price['others'].values()):
    #      ax.text(x, value1, '%.2f'%value1, ha='center', va='bottom', fontsize=6)
    #      ax.text(x, value2, '%.2f' % value2, ha='center', va='bottom', fontsize=6)
    #      ax.text(x, value3, '%.2f' % value3, ha='center', va='bottom', fontsize=6)

    plt.legend(loc=1)
    plt.show()
def av_price_player(df):
    player_av_price={}
    for name in ['AS','others']:
        data = df[df['player'] == name]
        market=list(set(data['market'].tolist()).tolist)
        market.sort()
        a = {}
        for s in market:
            price = data.loc[data['market'] == s]['price'].to_list()
            pax = data.loc[data['market'] == s]['pax'].to_list()
            av_price = np.average(price, weights=pax)
            a[s] = av_price
    player_av_price[name]=a

    return player_av_price

def pax_count(path,excel_name):
    dir = path
    os.chdir(dir)
    wb = xlrd.open_workbook(excel_name)
    sheets = wb.sheet_names()  # retun a list with all sheet names
    dic = {}
    for i,name  in enumerate(sheets):
        df = pd.read_excel(excel_name, sheet_name=name)
        pax_AS = df[df['player'] == 'AS']['pax'].sum()
        pax_OT = df[df['player'] == 'others']['pax'].sum()
        dic[i]=[pax_AS,pax_OT]
    return dic

def dict_to_mat(dict):
    """
    :param dict: 
    :return: 一个矩阵形式的 关于机场OD对的值信息
    """
    airports=list(set([eval(ap)[0] for ap in dict.keys()]))

    airports.sort(reverse=False)#进行排序 ANC在前
    # df = pd.DataFrame(index=pd.MultiIndex.from_product([airports,['Origin', 'Dest']]),columns=pd.MultiIndex.from_product([airports,['Origin', 'Dest']]))
    # for segemnt,value in dict.items():
    #     df.loc[(eval(segemnt)[0],'Origin' ), (eval(segemnt)[1],'Dest')]=value
    #     df.loc[(eval(segemnt)[1], 'Dest'), (eval(segemnt)[0], 'Origin')]=value
    df = pd.DataFrame(index=pd.MultiIndex.from_product([ ['Origin', 'Dest'],airports,]),
                      columns=pd.MultiIndex.from_product([ ['Origin', 'Dest'],airports,]))
    for segemnt,value in dict.items():
        df.loc[('Origin',eval(segemnt)[0] ), ('Dest',eval(segemnt)[1])]=value
        df.loc[( 'Dest',eval(segemnt)[1],), ( 'Origin',eval(segemnt)[0])]=value
    return df
if __name__ == "__main__":
    path='F:\jinyu visit\Dropbox\Pyomo_in_China\Pyomo3_smallPDX\data'
    os.chdir(path)
    #load_plot(path,'load_factor_compile.xlsx','title')
    df_price=pd.read_csv('av_price_beta0820.csv')
    current_result=pd.read_csv('current_result.csv')
    current_AS=pd.read_csv('AS_current_timetable_70.csv')
    current_ot = pd.read_csv('other_current_timetable35.csv')
    current_ot['UNIQUE_CARRIER'] = 'others'
    current_time_table=current_AS.append(current_ot)
    demand_data = pd.read_csv('keji_final.csv',)
    pre_dict_price=df_price.iloc[1,1:]
    dict_price=df_price.iloc[1,1:].to_dict()
    #av_price_player(current_result)
    segment_info = [(dept, arrive, demand, fre, ttime) for (dept, arrive, demand, fre, ttime) in
                    zip(demand_data['Dept'].tolist(), demand_data['Arrive'].tolist(),
                        demand_data['Demand'].tolist(), demand_data['frequency'].tolist(),
                        demand_data['travel_time'].tolist())]
    price_plot(current_result,df_price,current_time_table,'Current result',segment_info) #画图
    # df=dict_to_mat(dict_price)
    # df.to_csv('price_mat_display.csv')