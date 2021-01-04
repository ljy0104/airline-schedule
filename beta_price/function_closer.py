# -*- coding: utf-8 -*-
"""
Created on 2020/8/18  17:09 

# Author： Jinyu
"""
import pandas as pd
from concrete_model import set_model
from price_equi_solver import *
from assign_data_prepare import attr_value_compute_new,process_input_new,other_airline_iti
def result_to_df(player_obj_list):
    """
    this function can get obj and save obj to dataframe
    :param player_obj_list: 
    """
    data_list=[]
    flight_pax = []
    for player_obj in player_obj_list:
        for market_obj, itinerary_list in player_obj.market_itinerary.items():
            for i, iti_obj in enumerate(itinerary_list):
                for j, type in enumerate(market_obj.passenger_types_list):
                    a_dic = {'player': player_obj.name, 'market': market_obj.od_pair,
                             'depart_time': iti_obj.departure_time, \
                             'travel_time': iti_obj.travel_time, 'price': player_obj.itin2price[(iti_obj, type)],
                             'pax': player_obj.itin2pax[(iti_obj, type)],
                             'passenger_type': type, \
                             'denominator': market_obj.n_value, \
                             'demand': market_obj.market_demand_dic[type], 'connections': iti_obj.connection_ap, }
                    data_list.append(a_dic)

        for flight_obj in player_obj.flights:
            li = []
            d_time = []
            for itinerary_obj in flight_obj.itineraries:
                pax_iti = sum([v for k, v in player_obj.itin2pax.items() if k[0] == itinerary_obj])
                li.append(pax_iti)
                d_time.append(itinerary_obj.departure_time)
            pax = sum(li)
            f_dic = {'flight': flight_obj.flight_index, 'player': player_obj.name,'segment':flight_obj.segment, 'capacity': flight_obj.capacity,
                     'flight_pax': pax, \
                     'pax_per_iti': tuple(li), 'depart_time': d_time}
            flight_pax.append(f_dic)
    df_flight = pd.DataFrame(flight_pax)
    df_flight.eval('load_factor=flight_pax/capacity', inplace=True)
    df_iti = pd.DataFrame(data_list)



    return df_iti,df_flight

def current_timetable_price(tb_player1,tb_player_2,market_connections,market_instances_list):
    """
    :param pax_input: tuple of (type_name,proportions)
    :param tb_player1: 
    :param tb_player_2: 
    :param market_airports_AS: 
    :param market_airports_other: 
    :param market_demand: {('ANC', 'SEA'): ['PDX'], ('ANC', 'PDX'): ['SEA'], ('SEA', 'LAX'): ['PDX'],}
    """
    df_AS=pd.read_csv(tb_player1).sort_values(['ORIGIN','DEST'])
    df_other = pd.read_csv(tb_player_2).sort_values(['ORIGIN','DEST'])
    df_AS['FLIGHT_NUMBER']=['AS'+str(i) for i in range(1,len(df_AS['FLIGHT_NUMBER'])+1)]
    df_other['FLIGHT_NUMBER'] = ['OT' + str(i) for i in range(1, len(df_other['FLIGHT_NUMBER']) + 1)]

    un_order_AS_od_set = [(dept, arrive) for (dept, arrive) in zip(df_AS['ORIGIN'].tolist(), df_AS['DEST'].tolist())]
    un_order_other_od_set = [(dept, arrive) for (dept, arrive) in
                             zip(df_other['ORIGIN'].tolist(), df_other['DEST'].tolist())]
    AS_od_set = list(set(un_order_AS_od_set))
    AS_od_set.sort(key=un_order_AS_od_set.index)
    other_od_set = list(set(un_order_AS_od_set))
    other_od_set.sort(key=un_order_other_od_set.index)
    market_airports_AS = {key: market_connections[key] for key in AS_od_set}

    player_AS=Player("AS",1)
    #player_AS.df=df_all.loc[df_all['UNIQUE_CARRIER']=='AS']
    player_AS.df=df_AS
    player_other = Player('others', 2)
    player_other.df=df_other
    #generate itinerary and serve market
    iti_info_AS=player_AS.itinerary_generator(market_airports_AS,market_instances_list,30,120)
    iti_info_Others=player_other.itinerary_generator(market_airports_AS,market_instances_list,30,120)

    player_AS.itinerary=Itinerary.itinerary_instance_gen(iti_info_AS)#根据 time table 产生itinerary 的对象列表
    player_other.itinerary = Itinerary.itinerary_instance_gen(iti_info_Others)


    flight_AS = df_AS[['FLIGHT_NUMBER', 'ORIGIN','DEST','UNIQUE_CARRIER', 'Capacity', 'CRS_DEP_TIME']]
    flight_others = df_other[['FLIGHT_NUMBER', 'ORIGIN','DEST','UNIQUE_CARRIER', 'Capacity', 'CRS_DEP_TIME']]

    player_AS.flights=Flights.flight_instance_gen(flight_AS,'AS')
    player_other.flights=Flights.flight_instance_gen(flight_others,'other')
    player_AS.sequence_departure_generator()
    player_other.sequence_departure_generator()
    # create player dictionary of :{flight:[itinerary_index]}
    player_AS.flight_itinerary_record()
    player_other.flight_itinerary_record()
    for market_obj in market_instances_list:
        for player_obj in [player_AS,player_other]:
            if market_obj in player_obj.serve_market:
                market_obj.players.append(player_obj)
    D1 = Passenger_utility_mat_new()
    D1.compute(market_instances_list, [player_AS,player_other],-0.00387,-2.66,d_time_counted=True)
    game = Game(market_instances_list, [ player_AS,player_other])
    p_beta= -0.00678
    Strategies, prv, optimization_results, profits = game.solve_game_best_response(200,20,10,p_beta)
    df_iti, df_flight = result_to_df([player_AS, player_other])
    market_unset = df_iti['market'].to_list()
    market = list(set(market_unset))
    market.sort(key=market_unset.index)
    a = {}
    for s in market:
        price = df_iti.loc[df_iti['market'] == s]['price'].to_list()
        pax = df_iti.loc[df_iti['market'] == s]['pax'].to_list()
        av_price = np.average(price, weights=pax)
        a[s] = av_price
    return a, df_iti, df_flight, profits


def time_schedual_run(input_df_others,input_df,fleet_info,market_instances_list,passenger_proportions,passenger_peak_time,segment_travel_time,time_zone_dict,iti_dict,solver,):
    airport_list, market_data, iti_dict = process_input_new(input_df, 15, market_connections, segment_travel_time,
                                                            time_zone_dict)
    #airport_list = list(set(input_df['Dept'].tolist()))
    segment = [(dept, arrive) for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market = ['M' + str((s[0], s[1])) for s in segment]
    # travel_time=input_df['travel_time'].tolist()
    data = [(demand, miles, price, frequency, origin, dest) for (demand, miles, price, frequency, origin, dest) in
            zip(input_df['Demand'].tolist(), input_df['Nonstop_Miles'].tolist(), input_df['avg_price'].tolist(),
                input_df['frequency'].tolist(), input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_data = dict(zip(market, data))

    attr_value = attr_value_compute_new(iti_dict, 15, passenger_peak_time)
    marketpt_attr_sum = other_airline_iti(input_df_others, passenger_peak_time, no_fly_attr=0.2)
    # run pyomo model
    print(market_data)
    time_table, df_q, profit,cost,nonq_dict = set_model(airport_list, market_data, segment_travel_time,time_zone_dict, iti_dict, fleet_info,
                                         passenger_proportions, attr_value, marketpt_attr_sum,
                                         solver)
    return time_table,df_q,profit,cost,nonq_dict

def time_schedual_run_new(input_df_others,input_df,fleet_info,market_instances_list,passenger_proportions,passenger_peak_time,segment_travel_time,time_zone_dict,iti_dict,solver,):
    airport_list = list(set(input_df['Dept'].tolist()))
    segment = [(dept, arrive) for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market = ['M' + str((s[0], s[1])) for s in segment]
    # travel_time=input_df['travel_time'].tolist()
    data = [(demand, miles, price, frequency, origin, dest) for (demand, miles, price, frequency, origin, dest) in
            zip(input_df['Demand'].tolist(), input_df['Nonstop_Miles'].tolist(), input_df['avg_price'].tolist(),
                input_df['frequency'].tolist(), input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_data = dict(zip(market, data))

    attr_value = attr_value_compute_new(iti_dict, 15, passenger_peak_time)
    marketpt_attr_sum = other_airline_iti(input_df_others, passenger_peak_time, no_fly_attr=0.2)
    # run pyomo model
    print(market_data)
    time_table, df_q, profit,cost,nonq_dict = set_model(airport_list, market_data, segment_travel_time,time_zone_dict, iti_dict, fleet_info,
                                         passenger_proportions, attr_value, marketpt_attr_sum,
                                         solver)
    return time_table,df_q,profit,cost,nonq_dict

def time_schedual_run(input_df_others,input_df,fleet_info,market_instances_list,passenger_proportions,passenger_peak_time,solver,):
    airport_list, market_price,iti_dict = process_input_new(input_df, 15,market_instances_list)
    #airport_list = list(set(input_df['Dept'].tolist()))
    segment = [(dept, arrive) for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market = ['M' + str((s[0], s[1])) for s in segment]
    # travel_time=input_df['travel_time'].tolist()
    data = [(demand, miles, price, frequency, origin, dest) for (demand, miles, price, frequency, origin, dest) in
            zip(input_df['Demand'].tolist(), input_df['Nonstop_Miles'].tolist(), input_df['avg_price'].tolist(),
                input_df['frequency'].tolist(), input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_data = dict(zip(market, data))

    attr_value = attr_value_compute_new(iti_dict, 15, passenger_peak_time)
    marketpt_attr_sum = other_airline_iti(input_df_others, passenger_peak_time, no_fly_attr=0.2)
    # run pyomo model
    print(market_data)
    time_table, df_q, profit,cost,nonq_dict = set_model(market_instances_list,airport_list,market_price ,market_data, iti_dict, fleet_info,
                                        passenger_proportions, attr_value, marketpt_attr_sum,solver)

    return time_table,df_q,profit,cost,nonq_dict

def time_schedual_run_test(input_df_others,input_df,fleet_info,market_instances_list,passenger_proportions,passenger_peak_time,segment_travel_time,time_zone_dict,iti_dict,solver,):
    airport_list, market_data, iti_dict = process_input_new(input_df, 15, market_connections, segment_travel_time,
                                                            time_zone_dict)
    #airport_list = list(set(input_df['Dept'].tolist()))
    segment = [(dept, arrive) for (dept, arrive) in zip(input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market = ['M' + str((s[0], s[1])) for s in segment]
    # travel_time=input_df['travel_time'].tolist()
    data = [(demand, miles, price, frequency, origin, dest) for (demand, miles, price, frequency, origin, dest) in
            zip(input_df['Demand'].tolist(), input_df['Nonstop_Miles'].tolist(), input_df['avg_price'].tolist(),
                input_df['frequency'].tolist(), input_df['Dept'].tolist(), input_df['Arrive'].tolist())]
    market_data = dict(zip(market, data))

    attr_value = attr_value_compute_new(iti_dict, 15, passenger_peak_time)
    marketpt_attr_sum = other_airline_iti(input_df_others, passenger_peak_time, no_fly_attr=0.2)
    # run pyomo model
    print(market_data)
    time_table, df_q, profit,cost,nonq_dict = set_model(airport_list, market_data, segment_travel_time,time_zone_dict, iti_dict, fleet_info,
                                         passenger_proportions, attr_value, marketpt_attr_sum,
                                         solver)
    return time_table,df_q,profit,cost,nonq_dict
