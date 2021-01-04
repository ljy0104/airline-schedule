# -*- coding: utf-8 -*-
"""
Created on 2020/10/11  9:03 

# Author： Jinyu
"""
# -*- coding: utf-8 -*-
"""
Created on 2020/8/5  13:24 

# Author： Jinyu
"""
# -*- coding: utf-8 -*-
"""
Created on 2020/6/21  17:42 

# Author： Jinyu
"""

from assign_data_prepare import fleet_info_gen

from function_closer import *
from data_prepare import market_airports,new_price_replace,new_price_replace_others
from price_equi_solver import *

os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))
def print_obj(obj):
    "打印对象的所有属性"
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
def time_table_trans( df_time_table, segment_travel_time, fleet_name_capacity_dict, player_name ):
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
    for columns_name in df_time_table.columns[1:]:# 对行进行遍历，得到od pair
        for i in range(0, len(df_time_table)):
            fleet_type=df_time_table.loc[i][columns_name]
            if not pd.isnull(fleet_type):
                ORIGIN,DEST=columns_name[0],columns_name[1]
                CRS_ELAPSED_TIME=segment_travel_time[columns_name]
                df_new.loc[flight_index]=[flight_index, ((i-1)*15+360), ((i-1)*15+360) + CRS_ELAPSED_TIME, CRS_ELAPSED_TIME,player_name, ORIGIN, DEST, fleet_name_capacity_dict[fleet_type]]
                flight_index +=1
    return df_new

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
    df_keji = pd.read_csv('keji.csv')
    df_keji_others=pd.read_csv('other_current_timetable35.csv')
    #df_keji.loc[(df_keji['Dept'] == "SEA") & (df_keji['Arrive'] == "PDX"), 'actualdemand']=380
    #df_keji.loc[(df_keji['Dept'] == "PDX") & (df_keji['Arrive'] == "SEA"), 'actualdemand']=390
    solver = "gurobi"  # gurobi, glpk, etc.
    # set of fleet info , create 2 dicts: points_tulpe and fleet_info
    Fleet_name = ["737-400", "737-700", "737-800", "737-900", "737-400C"]
    AS_Fleet = [17, 11, 57, 44, 4]
    Capacity = [144, 124, 160, 180, 72]
    fleet_info, fleet_name_capacity_dict = fleet_info_gen(Fleet_name, AS_Fleet, Capacity)
    #rare data procession
    df_AS=pd.read_csv('AS_current_timetable.csv').sort_values(by='CRS_DEP_TIME')
    df_other = pd.read_csv('other_current_timetable.csv').sort_values(by='CRS_DEP_TIME')
    df_all=df_AS.append(df_other)
    df_all['FLIGHT_NUMBER']=range(1,len(df_all['FLIGHT_NUMBER'])+1)
    proportions = [0.248772702, 0.124024325, 0.171029853, 0.10838055, 0.213028019, 0.134764552]
    proportions.reverse()
    #用vikrant 计算的线性回归的参数，进行加总比值得出6类乘客的比例
    type_name = ['a', 'b', 'c', 'd', 'e', 'f']
    pax_input=(type_name,proportions)
    peak_time = [5, 8, 11, 14, 17, 20]
    passenger_peak_time = dict(zip(type_name, peak_time))
    passenger_proportions = dict(zip(type_name, proportions))

    market_connections=market_airports('keji')
    #segment=list(set([(ap1 ,ap2 ) for (ap1,ap2) in zip(df_all['ORIGIN'].tolist(),df_all['DEST'].tolist())]))
    market_demand={(ap1,ap2):demand for (ap1,ap2,demand) in zip(df_keji['Dept'].tolist(),df_keji['Arrive'].tolist(),df_keji['Demand'].tolist())}
    # av_price,result,profits=current_timetable_price('AS_current_timetable_70.csv','other_current_timetable35.csv', market_connections, market_demand,
    #                         pax_input)
    av_price ={('ANC', 'SEA'): 206.63325230849676, ('ANC', 'PDX'): 198.91730416929636,
                  ('SEA', 'LAX'): 208.42547668844983, ('LAX', 'SEA'): 208.3431678361646,
                  ('LAX', 'PDX'): 202.80071427634303, ('SEA', 'ANC'): 206.2490282550707,
                  ('PDX', 'LAX'): 203.27718150864268, ('PDX', 'SEA'): 209.68629988311693,
                  ('PDX', 'ANC'): 197.84575827492563, ('SEA', 'PDX'): 232.09689056877465,
                  ('ANC', 'LAX'): 200.65643407593262, ('LAX', 'ANC'): 200.63409627727756}
    input_df = new_price_replace(df_keji, av_price)
    input_df_others=new_price_replace_others(df_keji_others,av_price)
    #result.to_csv('current_result.csv')
    print('av_price is:', av_price)
    #print("current profit is :", profits)
    data_list=[]
    l=[av_price]
    iteration=1
    for i in range(iteration):
        print('this is iteration :', i)
        # input_df=new_price_replace(input_df,market_price)
        airport_list, market_data, segment_travel_time, iti_dict = process_input_new(input_df, 15)
        time_table, df_q, profit, cost = time_schedual_run(input_df_others,fleet_info, passenger_proportions,
                                                                      passenger_peak_time, solver, input_df,
                                                                      time_step=15)
        time_table.to_csv("assignment_table" + str(i) + ".csv")
        df_AS = time_table_trans(time_table, segment_travel_time, fleet_name_capacity_dict, 'AS')
        df_AS.to_csv('timetable' + str(i) + '.csv')
        print('finish' * 5)
        new_price,result_df,profit_equi= current_timetable_price('timetable' + str(i) + '.csv', 'other_current_timetable35.csv',
                                           market_connections, market_demand,pax_input)
        result_df.to_csv('result_df' + str(i) + '.csv')
        count = 1

        print("new_av_price:", new_price)
        print("two player profit is :", profit_equi)
        # print("res_ proess is  :", optimization_results)
        input_df = new_price_replace(input_df, new_price)
        count += 1
        #print(input_df['avg_price'])  # market_price=player_market_avprice['AS']
        sub_data_list = [61,profit,cost,profit_equi[1],profit_equi[2]]
        data_list.append(sub_data_list)
        l.append(new_price)
    m_index = pd.MultiIndex.from_product([range(iteration), ['gap', 'profit','cost','player_1_equi','player_2_equi']], names=["iterations", "class2"])
    df3 = pd.DataFrame(data_list, columns= ['gap', 'profit','cost','player_1_equi','player_2_equi'])
    df3.to_csv('iteration_result.csv')
    price_df = pd.DataFrame(l)
    price_df.to_csv('av_price_beta0820.csv')


