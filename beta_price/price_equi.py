# -*- coding: utf-8 -*-
# encoding: utf-8
"""
Created on 2020/8/5  13:24 

# Author： Jinyu
"""
# -*- coding: utf-8 -*-

'demand 要全用demand 而不用actualdemand' \

"""
Created on 2020/6/21  17:42 

# Author： Jinyu
"""

from assign_data_prepare import fleet_info_gen
from function_closer import *
from data_prepare import market_airports,new_price_replace,new_price_replace_others,day_preference_curves,market_preferred_time,TODS_decide
from price_equi_solver import *
import pandas as pd
os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))
# def print_obj(obj):
#     "打印对象的所有属性"
#     print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
def time_table_trans( df_time_table, segment_travel_time,segment_time_zone, fleet_name_capacity_dict, player_name ):
    """
    :param df_time_table: df from keji's model ,time_table as fleet assiginment
    :param segment_travel_time: dict with key (OD_pair) and value travel time
    :param fleet_name_capacity_dict: fleet name : capacity
    :param player_name: which player (str)
    :return: time table with departure time.... as 
    """
    #create a empty dataframe
    df_new = pd.DataFrame(columns=["FLIGHT_NUMBER", "CRS_DEP_TIME","CRS_ARR_TIME","CRS_ELAPSED_TIME","UNIQUE_CARRIER","ORIGIN","DEST","Capacity"])
    flight_index =1
    for columns_name in df_time_table.columns:# 对行进行遍历，得到od pair
        for i in range(len(df_time_table)):
            fleet_type=df_time_table.iloc[i][columns_name]
            if not pd.isnull(fleet_type):
                ORIGIN,DEST=columns_name[0],columns_name[1]
                CRS_ELAPSED_TIME=segment_travel_time[columns_name]
                time_zone_diff=segment_time_zone[columns_name]
                df_new.loc[flight_index]=[flight_index, ((i-1)*15+360), ((i-1)*15+360) + CRS_ELAPSED_TIME+time_zone_diff, CRS_ELAPSED_TIME,player_name, ORIGIN, DEST, fleet_name_capacity_dict[fleet_type]]
                flight_index +=1
    df_new.loc[df_new['CRS_ARR_TIME'] >1440, 'CRS_ARR_TIME'] = df_new['CRS_ARR_TIME'] - 1440

    return df_new

def time_zone(str_name):
    "返回一个字典，key是两个机场，value是之间的time_different."
    #airport_list = list(set(df['ORIGIN'].tolist()))
    df=pd.read_csv(str_name)
    segment = list(set([(dept, arrive) for (dept, arrive) in zip(df['ORIGIN'].tolist(), df['DEST'].tolist())]))
    dict_tz={}
    for s in segment:
        a=df[(df['ORIGIN']==s[0])&(df['DEST']==s[1])].iloc[0]
        if (a['CRS_ARR_TIME'] - a['CRS_DEP_TIME'])>=0:
            diff=(a['CRS_ARR_TIME']-a['CRS_DEP_TIME'])-a['Flying Time']
        else:
            diff=(1440-a['CRS_DEP_TIME']+a['CRS_ARR_TIME'])-a['Flying Time']
        dict_tz[s]=diff
    return dict_tz

def time_swich(number):
    str = repr(number)
    if len(str)<=2:
        min_time_format=int(str)
    else:
        numN1 = int(str[-2:])
        numN2 =int(str[:-2])
        min_time_format=numN2*60+numN1
    return min_time_format
if __name__ == "__main__":
    df_keji = pd.read_csv('keji_final.csv')
    df_keji_others=pd.read_csv('other_current_timetable35.csv')
    segment = [(dept, arrive) for (dept, arrive) in zip(df_keji['Dept'].tolist(), df_keji['Arrive'].tolist())]
    travel_time = df_keji['travel_time'].tolist()
    segment_travel_time = dict(zip(segment, travel_time))
    solver = "gurobi"  # gurobi, glpk, etc.
    # set of fleet info , create 2 dicts: points_tulpe and fleet_info
    Fleet_name = ["737-400", "737-700", "737-800", "737-900", "737-400C"]
    AS_Fleet = [17, 11, 57, 44, 4]
    Capacity = [144, 124, 160, 180, 72]
    fleet_info, fleet_name_capacity_dict = fleet_info_gen(Fleet_name, AS_Fleet, Capacity)
    #乘客信息部分 ：包括名字，比例
    proportions = [0.248772702, 0.124024325, 0.171029853, 0.10838055, 0.213028019, 0.134764552]
    proportions.reverse()
    #用vikrant 计算的线性回归的参数，进行加总比值得出6类乘客的比例
    type_name = ['a', 'b', 'c', 'd', 'e', 'f']
    #pax_input=(type_name,proportions)
    peak_time = [5, 8, 11, 14, 17, 20]
    passenger_peak_time = dict(zip(type_name, peak_time))
    passenger_proportions = dict(zip(type_name, proportions))

    time_zone_dict=time_zone('AS_current_timetable_60.csv')
    market_connections=market_airports('keji')

    market_demand_distance={(ap1,ap2):(demand,distance) for (ap1,ap2,demand,distance) in zip(df_keji['Dept'].tolist(),df_keji['Arrive'].tolist(),df_keji['Demand'].tolist(),df_keji['Nonstop_Miles'].tolist())}
    market_distance={(ap1,ap2):demand for (ap1,ap2,demand) in zip(df_keji['Dept'].tolist(),df_keji['Arrive'].tolist(),df_keji['Nonstop_Miles'].tolist())}

    market_instances_list = Market.market_instance_gen(market_demand_distance)
    for m in market_instances_list:
        m.mistiming=time_zone_dict[m.od_pair]
        m.connections=market_connections[m.od_pair]
        m.travel_time=segment_travel_time[m.od_pair]
        TODS_number=TODS_decide(m.mistiming,m.distance)
        Y=market_preferred_time(TODS_number,1,"parames_of_departure_types.xlsx",aggregate_ratio=[0.1,0.45,0.45])
        proportions=day_preference_curves(Y,passenger_peak_time,3)
        for name,pr,peaktime in zip(passenger_peak_time.keys(),proportions,passenger_peak_time.values()):
            m.add_pax_obj(name,pr,peaktime)
        m.add_passenger()
        if m.od_pair==('SEA', 'PDX')or m.od_pair==( 'PDX','SEA'):
            m.n_value=0.2
        else:
            m.n_value=0.2

    # av_price,result,df_flight,profits=current_timetable_price('AS_current_timetable_60.csv','other_current_timetable35.csv', market_connections, market_instances_list,)
    # result.to_csv('current_result.csv')
    # df_flight.to_csv('current_loadfactor.csv')
    av_price= {('ANC', 'SEA'): 302.24320710647817, ('ANC', 'LAX'): 214.841956012637,
                   ('ANC', 'PDX'): 232.09499139637717, ('LAX', 'PDX'): 341.95680392987754,
                   ('LAX', 'SEA'): 308.22074523270845, ('LAX', 'ANC'): 220.70638106384285,
                   ('PDX', 'LAX'): 286.7759679103416, ('PDX', 'SEA'): 301.5540061235026,
                   ('PDX', 'ANC'): 255.19758577961153, ('SEA', 'LAX'): 438.692686648481,
                   ('SEA', 'PDX'): 410.2994987730045, ('SEA', 'ANC'): 234.59895807559516}
    input_df = new_price_replace(df_keji, av_price)
    input_df_others=new_price_replace_others(df_keji_others,av_price)
    print('av_price is:', av_price)
    #print("current profit is :", profits)
    data_list=[]
    l=[av_price]
    iteration=10
    nonq_list_resort=[]
    airport_list, market_price,iti_dict=process_input_new(input_df, 15, market_instances_list,)
    #airport_list,market_data, iti_dict=process_input_new(input_df, 15,market_connections,segment_travel_time,time_zone_dict,market_instances_list)
    for i in range(iteration):
        print('this is iteration :', i)
        time_table, q_data, profit, cost, nonq_dict = time_schedual_run(input_df_others, input_df,fleet_info,market_instances_list,passenger_proportions,
                                                                        passenger_peak_time,solver,)
        #input_df_others, input_df, fleet_info, market_instances_list, passenger_proportions, passenger_peak_time, solver,
        time_table.to_csv("assignment_table" + str(i) + ".csv")
        nonq_list_resort.append(pd.Series(nonq_dict.values(),index=nonq_dict.keys()))
        #time_table=pd.read_csv('assignment_table0.csv')
        df_AS = time_table_trans(time_table, segment_travel_time,time_zone_dict, fleet_name_capacity_dict, 'AS')
        df_AS.to_csv('timetable' + str(i) + '.csv')
        print('finish' * 5)
        new_price,result_df,df_flight,profit_equi= current_timetable_price('timetable' + str(i) + '.csv', 'other_current_timetable35.csv',
                                           market_connections, market_instances_list)
        df_flight.to_csv('df_flight'+ str(i) + '.csv')
        result_df.to_csv('result_df' + str(i) + '.csv')
        count = 1

        print("new_av_price:", new_price)
        print("two player profit is :", profit_equi)
        # print("res_ proess is  :", optimization_results)
        input_df = new_price_replace(input_df, new_price)
        count += 1
        #print(input_df['avg_price'])  # market_price=player_market_avprice['AS']
        sub_data_list = [61,profit,cost,sum(q_data.values()),profit_equi[1],profit_equi[2]]
        data_list.append(sub_data_list)
        l.append(new_price)


    #处理并储存数据了
    pd.concat(nonq_list_resort, axis=1).to_csv('nonq_player.csv')
    m_index = pd.MultiIndex.from_product([range(iteration), ['gap', 'profit','cost','player_1_pax','player_1_equi','player_2_equi']], names=["iterations", "class2"])
    df3 = pd.DataFrame(data_list, columns= ['gap', 'profit','cost','player_1_pax','player_1_equi','player_2_equi'])
    df3.to_csv('iteration_result.csv')
    price_df = pd.DataFrame(l)
    price_df.to_csv('av_price_beta0820.csv')


