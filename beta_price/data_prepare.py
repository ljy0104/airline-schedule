# -*- coding: utf-8 -*-
"""
Created on 2020/6/18  13:32 

# Author： Jinyu
"""
import numpy as np
import math
import os
import pandas as pd
import math
from scipy.stats import norm
from sklearn import datasets,linear_model

os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))

def attractiveness_old(d_time,e_time,price,elapsed_time_para,sc_fun_list,beta,connections):
    #a=[-0.016,- 0.041,0.014, - 0.115, - 0.184, - 0.053]
    #b=[0.211,-0.270,-0.134,-0.996,-0.500,0.030]
    sc_fun_value = np.array([np.sin((2 * d_time * math.pi) / 1440), np.sin((4 * d_time * math.pi) / 1440),
                       np.sin((6 * d_time * math.pi) / 1440),
                       np.cos((2 * d_time * math.pi) / 1440), np.cos((4 * d_time * math.pi) / 1440),
                       np.cos((6 * d_time * math.pi) / 1440)])
    depart_value = np.dot(np.array([sc_fun_list]), sc_fun_value)
    travel_time_value=elapsed_time_para*e_time
    p_value=price*beta
    attractive_value = np.exp(depart_value + travel_time_value +p_value+ connections * -2.66)[0]

    return attractive_value

def process_input(input_file):
    #read file, formatted as example file 'mentor_signups_2018.csv', skipping row zero
    input_df = pd.read_csv(input_file + ".csv")
    #get number of airports and data
    airport_list=list(set(input_df['Dept'].tolist()))
    segment=[(dept, arrive)for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market = ['M' + str((s[0], s[1])) for s in segment]
    travel_time=input_df['travel_time'].tolist()
    data = [(demand, miles, price, frequency,origin,dest) for (demand, miles, price, frequency,origin,dest) in
            zip(input_df['Demand'].tolist(), input_df['Nonstop_Miles'].tolist(), input_df['avg_price'].tolist(),
                input_df['frequency'].tolist(),input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_data=dict(zip(market,data))

    segment_travel_time=dict(zip(segment,travel_time))

    market_airports = {}
    for i, s in enumerate(segment):
        ap_connection = []
        for ap in airport_list:
            if ap != s[0] and ap != s[1]:
                distance_1 = input_df.loc[(input_df['Dept'] == s[0]) & (input_df['Arrive'] == ap)]['Nonstop_Miles'].tolist()[0]
                distance_2 = input_df.loc[(input_df['Dept'] == ap) & (input_df['Arrive'] == s[1])]['Nonstop_Miles'].tolist()[0]
                nonstop_distance = input_df.loc[(input_df['Dept'] == s[0]) & (input_df['Arrive'] == s[1])]['Nonstop_Miles'].tolist()[0]
                if (distance_1 + distance_2) <= nonstop_distance * 1.5:
                    ap_connection.append(ap)
        market_airports.update({s: ap_connection})
    non_stop = []
    one_stop = []
    for key in segment:
        market = 'M' + str((key[0], key[1]))
        for t in range(1,73):
            non_stop.append({'market': market, 'non_stop': (key[0], key[1]), (key[0], key[1]): t,'legs':[(key[0], key[1])]})
        if market_airports[key]:
            for ap in market_airports[key]:
                t_time = \
                input_df.loc[(input_df['Dept'] == key[0]) & (input_df['Arrive'] == ap)]['travel_time'].tolist()[0]
                for t1 in range(1, 73):
                    for t2 in range(1, 73):
                        if t1 + (t_time // 15) + 1 < t2<t1 + (t_time // 15)+4:
                            one_stop.append({'market': market, 'non_stop': None, 'first_leg': (key[0], ap), (key[0], ap): t1,
                                             'second_leg': (ap, key[1]), (ap, key[1]): t2,'legs':[(key[0], ap),(ap, key[1])]})
    record = non_stop + one_stop
    iti_dict = {i: value for i, value in enumerate(record)}
    return airport_list,market_data,segment_travel_time,iti_dict

def read_departure_para(file_name):
    data_xls = pd.ExcelFile(file_name)
    data = {}
    #print(data_xls.sheet_names)
    for name in data_xls.sheet_names:
        df = data_xls.parse(name,index_col=0)
        data[name] = df
    return data

def market_airports(input_file):
    input_df = pd.read_csv(input_file + ".csv")
    # get number of airports and data
    airport_list = list(set(input_df['Dept'].tolist()))
    segment = [(dept, arrive) for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_airports = {}
    for i, s in enumerate(segment):
        ap_connection = []
        for ap in airport_list:
            if ap != s[0] and ap != s[1]:
                distance_1 = \
                input_df.loc[(input_df['Dept'] == s[0]) & (input_df['Arrive'] == ap)]['Nonstop_Miles'].tolist()[0]
                distance_2 = \
                input_df.loc[(input_df['Dept'] == ap) & (input_df['Arrive'] == s[1])]['Nonstop_Miles'].tolist()[0]
                nonstop_distance = \
                input_df.loc[(input_df['Dept'] == s[0]) & (input_df['Arrive'] == s[1])]['Nonstop_Miles'].tolist()[0]
                if (distance_1 + distance_2) <= nonstop_distance * 1.5:
                    ap_connection.append(ap)
        market_airports.update({s: ap_connection})
    return market_airports

def keji_to_csv(input_file):
    input_df = pd.read_csv(input_file + ".csv")
    df_new = pd.DataFrame(columns=["FLIGHT_NUMBER", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "UNIQUE_CARRIER", "ORIGIN","DEST", "Capacity"])
    pass
def new_price_replace(df,dic):
    """
    this function is for repalce av_price from new price in keji.csv. 
    :param df: old csv
    :param dic: dic is like {(ap1,ap2):price}
    :return: 
    """
    for od_pair,price in dic.items():
        df.loc[(df['Dept'] == od_pair[0]) & (df['Arrive'] ==od_pair[1]), 'avg_price']=price

    return df

def new_price_replace_others(df,dic):
    for od_pair, price in dic.items():
        df.loc[(df['ORIGIN'] == od_pair[0]) & (df['DEST'] == od_pair[1]), 'avg_price'] = price
    return df
def get_min_load_factor_flight(self,player_name,jump=False):
     flight_pax = []
     for player_obj in self.players:
         for flight_obj in player_obj.flights:
             li = []
             d_time = []
             for itinerary_obj in flight_obj.itineraries:
                 pax_iti = sum([v for k, v in player_obj.itin2pax.items() if k[0] == itinerary_obj])
                 li.append(pax_iti)
                 d_time.append(itinerary_obj.departure_time)
             pax = sum(li)
             f_dic = {'flight': flight_obj.flight_index, 'player': player_obj.name, 'capacity': flight_obj.capacity,
                      'flight_pax': pax, \
                      'pax_per_iti': tuple(li), 'depart_time': d_time}
             flight_pax.append(f_dic)
     df1 = pd.DataFrame(flight_pax)
     df1.eval('load_factor=flight_pax/capacity', inplace=True)
     self.flight_result_info=df1
     df1_player=df1.loc[df1['player'] == player_name].sort_values('load_factor')
     if not jump:
        self.chosen_flight_index=df1_player.iloc[0,:]['flight']# {player_obj:flight_index}
     else:
         self.chosen_flight_index = df1_player.iloc[1,:]['flight']


def operation_cost(tuple_list_AS):
    v=[]
    for i in tuple_list_AS:
        if i[0] <= 3106:
            value=(1.6 *i[0]  + 722) * (i[1] + 104) * 0.019
        else:
            value =(1.6 * i[0] + 2200) * (i[1] + 211) * 0.0115
        v.append(value)
    cost=sum(v)
    return cost

def day_preference_curves(aggregate,passenger_peak_time,Std_Dev):
    #首先建立 6个pt的对于4点至23点的departure preference
    d_time=list(range(4,24))
    utility_mat = np.zeros(( len(d_time),len(passenger_peak_time)))
    for i,t in enumerate(passenger_peak_time.values()):
        utility_mat[:,i] = norm(t,Std_Dev).pdf(d_time)
    Y=aggregate
    X= utility_mat
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    coef=regr.coef_#Model.intercept_  查看截距
    pax_propertion=coef/sum(coef)

    return pax_propertion

def market_preferred_time(TODS_number,daytime_number,para_xls_name,aggregate_ratio=[0.1,0.45,0.45]):

    df = pd.read_excel(para_xls_name,sheet_name=TODS_number,index_col=0)

    para_mat = np.array(df.iloc[daytime_number - 1:daytime_number + 2])
    time=np.array(range(4,24))*60
    scfun = np.array([np.sin((2*time* math.pi) / 1440), np.sin((4 *time* math.pi) / 1440), np.sin((6*time * math.pi) / 1440),
                      np.cos((2 *time* math.pi) / 1440), np.cos((4 *time* math.pi) / 1440), np.cos((6*time * math.pi) / 1440)])

    depart_passenger_mat=np.dot(para_mat,scfun)
    aggregate=np.sum([depart_passenger_mat[i]*aggregate_ratio[i] for i in range (len(aggregate_ratio))],axis=0)
    return  aggregate

def TODS_decide(time,distance):
    if distance <= 600:
        d='short'
    else:
        d='long'
    result={(0,'short'):'TODS1',(0,'long'):'TODS2',(-60,'short'):'TODS3',(-60,'long'):'TODS4',(60,'short'):'TODS5',
            (60, 'long'): 'TODS6', (-120, 'long'): 'TODS7', (120, 'long'): 'TODS8', (-180, 'long'): 'TODS9',
            (1800, 'long'): 'TODS10'}
    return  result[(time,d)]
