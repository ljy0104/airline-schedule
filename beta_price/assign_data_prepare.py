# -*- coding: utf-8 -*-
"""
Created on 2020/7/16  15:50 

# Author： Jinyu
"""
# -*- coding: utf-8 -*-
"""
Created on 2020/6/18  13:32 

# Author： Jinyu
"""

import numpy as np
import math
import os
import pandas as pd
from scipy.stats import norm
os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))

def attractiveness(passenger_peak,d_time,e_time,price,elapsed_time_para,beta,connections,Std_Dev):
    """
    this function compute arrtactiveness of itinerary with base their d_time,t_time .... 
    for different types of passengers
    :param passenger_peak: passenger ideal time
    :param d_time: 
    :param e_time: 
    :param price: 
    :param elapsed_time_para: 
    :param sc_fun_list: 
    :param beta: price coefficient
    :param connections: 
    :return: 
    """
    depart_value = norm(passenger_peak, Std_Dev).pdf(d_time/60)
    travel_time_value=elapsed_time_para*e_time
    p_value=price*beta
    attractive_value = np.exp(depart_value + travel_time_value +p_value+ connections * -2.66)

    return attractive_value

def other_airline_iti(input_file,passeger_type_dict,no_fly_attr):
    input_df = input_file
    airport_list = list(set(input_df['DEST'].tolist()))
    segment=[(i, j) for i in airport_list for j in airport_list if i != j]
    market_name = ['M' + str((s[0], s[1])) for s in segment]
    other_market_attr={}
    for i in range(len(input_df)):
        d_time = input_df.loc[i]['CRS_DEP_TIME']
        e_time = input_df.loc[i]['CRS_ELAPSED_TIME']
        price = input_df.loc[i]['avg_price']
        connections = 0
        market='M'+str((input_df.loc[i]['ORIGIN'],input_df.loc[i]['DEST']))
        for type ,peak in passeger_type_dict.items():
            a_value = attractiveness(peak, d_time, e_time, price, -0.00387, -0.00657, connections,Std_Dev=3)
            other_market_attr[(i,type)]=[market,a_value]
        marketpt_attr_sum={}
    for m in market_name:
        for type in passeger_type_dict.keys():
            a = [value[1] for key, value in other_market_attr.items() if key[1]==type and  value[0] == m]
            on_fly=no_fly_attr
            marketpt_attr_sum[(m, type)] = sum(a)+on_fly
    return marketpt_attr_sum

def process_input(input_file,time_step):
    #this function will give basic list and dict which will used in pyomo_model from input csv file ,
    """
    :param input_file: csv name
    :param time_step: time_space of two adjacent departure time options
    :return: airport_list['LAX', 'ANC', 'PDX', 'SEA']
                market_data: dict with key as market,value as :[info]
                segment_travel_time
                iti_dict: key number of itineary value:[info]
                market_airports: a dict as key is od_pair ,value is list of ap
    """
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
        for t in range(1,(1440+time_step-360)//time_step):
            non_stop.append({'market': market, 'non_stop': (key[0], key[1]), (key[0], key[1]): t,'legs':[(key[0], key[1])],
                             'price':market_data[market][2],'travel_time':segment_travel_time[(key[0], key[1])]})
        if market_airports[key]:
            for ap in market_airports[key]:
                t_time = \
                input_df.loc[(input_df['Dept'] == key[0]) & (input_df['Arrive'] == ap)]['travel_time'].tolist()[0]
                for t1 in range(1, (1440+time_step-360)//time_step):
                    for t2 in range(1, (1440+time_step-360)//time_step):
                        if t1 + (t_time // time_step) + 2 < t2<t1 + (t_time // time_step)+8:
                            one_stop.append({'market': market, 'non_stop': None, 'first_leg': (key[0], ap), (key[0], ap): t1,
                                             'second_leg': (ap, key[1]), (ap, key[1]): t2,'legs':[(key[0], ap),(ap, key[1])],
                                             'price': market_data[market][2]*0.8,'travel_time':(t2-t1)*15+segment_travel_time[(ap, key[1])]})
    record = non_stop + one_stop
    iti_dict = {i: value for i, value in enumerate(record)}

    return airport_list,market_data,segment_travel_time,iti_dict,market_airports


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

def attr_value_compute_new(iti_dict,time_step,passeger_type_dict):
    attr_value = {}
    for k, v in iti_dict.items():
        if v['non_stop']:
            d_time = (v[v['non_stop']] - 1) *time_step + 360
            e_time = v['travel_time']
            price = v['price']
            connections = 0
        else:
            d_time = (v[v['first_leg']] - 1) * time_step + 360
            e_time = v['travel_time']
            price = v['price']
            connections = 1
        for type,peak in passeger_type_dict.items():
            a_value = attractiveness(peak,d_time, e_time, price, -0.00387, -0.00657, connections,Std_Dev=3)
            attr_value[(k, type)] = a_value
    return  attr_value

def process_input_old(input_df,time_step,market_connections,segment_travel_time,time_zone_dict,market_instances_list):
    #this function will give basic list and dict which will used in pyomo_model from input csv file ,
    """
    :param input_file: csv name
    :param time_step: time_space of two adjacent departure time options
    :return: airport_list['LAX', 'ANC', 'PDX', 'SEA']
                market_data: dict with key as market,value as :[info]
                segment_travel_time
                iti_dict: key number of itineary value:[info]
                market_airports: a dict as key is od_pair ,value is list of ap
    """
    market_instances_list=market_instances_list
    airport_list=list(set(input_df['Dept'].tolist()))
    segment=[(dept, arrive)for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market = ['M' + str((s[0], s[1])) for s in segment]
    #travel_time=input_df['travel_time'].tolist()
    data = [(demand, miles, price, frequency,origin,dest) for (demand, miles, price, frequency,origin,dest) in
            zip(input_df['Demand'].tolist(), input_df['Nonstop_Miles'].tolist(), input_df['avg_price'].tolist(),
                input_df['frequency'].tolist(),input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_data=dict(zip(market,data))
    non_stop = []
    one_stop = []
    for key in segment:
        market = 'M' + str((key[0], key[1]))
        for t in range(1,(1440+time_step-360)//time_step):
            non_stop.append({'market': market, 'non_stop': (key[0], key[1]), (key[0], key[1]): t,'legs':[(key[0], key[1])],
                             'price':market_data[market][2],'travel_time':segment_travel_time[(key[0], key[1])]})
        if market_connections[key]:
            for ap in market_connections[key]:
                t_time = segment_travel_time[( key[0],ap)]
                time_diff=time_zone_dict[( key[0],ap)]
                for t1 in range(1, (1440+time_step-360)//time_step):
                    for t2 in range(1, (1440+time_step-360)//time_step):
                        if t1 + ((t_time+time_diff) // time_step) + 2 < t2<t1 + ((t_time+time_diff) // time_step)+8:
                            one_stop.append({'market': market, 'non_stop': None, 'first_leg': (key[0], ap), (key[0], ap): t1,
                                             'second_leg': (ap, key[1]), (ap, key[1]): t2,'legs':[(key[0], ap),(ap, key[1])],
                                             'price': market_data[market][2]*0.8,'travel_time':(t2-t1)*15+segment_travel_time[(ap, key[1])]})
    record = non_stop + one_stop
    iti_dict = {i: value for i, value in enumerate(record)}

    return airport_list,market_data,iti_dict


def fleet_info_gen(Fleet_name,Fleet_number_list, Capacity_list):
    points_tulpe = list(zip(Fleet_number_list, Capacity_list))
    fleet_info = dict(zip(Fleet_name, points_tulpe))
    fleet_name_capacity_dict = dict(zip(Fleet_name, Capacity_list))
    return fleet_info,fleet_name_capacity_dict

def process_input_new(input_df,time_step,market_instances_list,):
    #this function will give basic list and dict which will used in pyomo_model from input csv file ,
    """
    :param input_file: csv name
    :param time_step: time_space of two adjacent departure time options
    :return: airport_list['LAX', 'ANC', 'PDX', 'SEA']
                market_data: dict with key as market,value as :[info]
                segment_travel_time
                iti_dict: key number of itineary value:[info]
                market_airports: a dict as key is od_pair ,value is list of ap
    """
    segment_travel_time={m.od_pair:m.travel_time for m in market_instances_list}
    time_zone_dict={m.od_pair:m.mistiming for m in market_instances_list}
    airport_list=list(set(input_df['Dept'].tolist()))
    #segment=[(dept, arrive)for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_price={m.od_pair:input_df.loc[(input_df["Dept"] == m.od_pair[0]) & (input_df["Arrive"] == m.od_pair[1])]['avg_price'].tolist()[0] for m in market_instances_list}

    non_stop1 = []
    one_stop1 = []
    for m_obj in market_instances_list:
        m_obj.iti_ind=[]#this step is for next pyomo model
        market = 'M' + str(m_obj.od_pair)
        for t in range(1,(1440+time_step-360)//time_step):
            non_stop1.append({'market': market, 'non_stop': (m_obj.od_pair[0],m_obj.od_pair[1]), (m_obj.od_pair[0],m_obj.od_pair[1]): t,'legs':[(m_obj.od_pair[0], m_obj.od_pair[1])],
                             'price':market_price[m_obj.od_pair],'travel_time':m_obj.travel_time})
        if m_obj.connections:
            for ap in m_obj.connections:
                t_time = segment_travel_time[(m_obj.od_pair[0],ap)]
                time_diff=time_zone_dict[(m_obj.od_pair[0],ap)]
                for t1 in range(1, (1440+time_step-360)//time_step):
                    for t2 in range(1, (1440+time_step-360)//time_step):
                        if t1 + ((t_time+time_diff) // time_step) + 2 < t2<t1 + ((t_time+time_diff) // time_step)+8:
                            one_stop1.append({'market': market, 'non_stop': None, 'first_leg': (m_obj.od_pair[0], ap), (m_obj.od_pair[0], ap): t1,
                                             'second_leg': (ap, m_obj.od_pair[1]), (ap, m_obj.od_pair[1]): t2,'legs':[(m_obj.od_pair[0], ap),(ap, m_obj.od_pair[1])],
                                             'price': market_price[m_obj.od_pair]*0.8,'travel_time':(t2-t1)*15+segment_travel_time[(ap, m_obj.od_pair[1])]})
    record = non_stop1 + one_stop1
    iti_dict = {i: value for i, value in enumerate(record)}

    return airport_list,market_price,iti_dict