# -*- coding: utf-8 -*-
"""
Created on 2020/1/28  19:24 

# Author： Jinyu
"""
import math
# from ipopt import minimize_ipopt
import os
import time

# import matplotlib.pyplot as plt
import autograd.numpy as np
import numpy as nu
from autograd import grad
from scipy.optimize import minimize
from scipy.stats import norm

os.chdir(os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'data')))

class Player(object):
    def __init__(self, name,player_index):
        self.name= name
        self.player_index=player_index
        self.serve_market=[] #need input by hand
        self.flights = [] #flights_obj player have
        self.itinerary=None #full info of itinerary_obj
        self.itinerary_number=0
        self.flights_number = 0
        self.df=None
        self.market_itinerary={}#{market_obj:[list of itinerary_obj]
        self.market_itinerary_dtime={}
        self.market_itinerary_ttime={}

        self.flight_itinerary_dic={} #{flight_carrier_obj:[index of itinerary in self.itinerary]}
        self.strategy_vector_size = None

        self.attra_probability_list=[]

        self.itin2price_ind = {}
        self.itin2pax_ind = {}
        # self.price_ind2itin = {}
        # self.pax_ind2itin = {}
        self.itin2price = {}
        self.itin2pax={}
        self.profit='none'
    def __repr__( self ):
        return 'name :{},player_index{}, '.format(self.name,self.player_index)
    def itinerary_generator(self,market_airports,market_obj_list,delta,max_connection_time):
        leg_1 = []
        connection = []
        serve_market=[]
        for od_pair,airport_list in market_airports.items():
            market_obj=[obj for obj in market_obj_list if obj.od_pair ==od_pair][0]
            #serve_market.append(market_obj)
            data_in = self.df.loc[(self.df['ORIGIN'] ==od_pair[0]) & (self.df['DEST'] == od_pair[1])].sort_values(by='CRS_DEP_TIME')
            for i in range(len(data_in)):
                leg_1.append({'first_flight': data_in.iloc[i]['FLIGHT_NUMBER'], 'second_flight': 'none', \
                              'departure_time': data_in.iloc[i]['CRS_DEP_TIME'], \
                              'travel_time': data_in.iloc[i]['CRS_ELAPSED_TIME'], \
                              'market': od_pair, 'connection_airport': 'None','market_obj':market_obj})
                serve_market.append(market_obj)
            if airport_list:
                for ap in airport_list:
                    data_in=self.df.loc[(self.df['ORIGIN'] ==od_pair[0]) & (self.df['DEST'] == ap)].sort_values(by='CRS_DEP_TIME')
                    data_con = self.df.loc[(self.df['ORIGIN'] == ap) & (self.df['DEST'] ==od_pair[1])].sort_values(by='CRS_DEP_TIME')
                    time_in= data_in['CRS_DEP_TIME'].tolist()
                    time_con=data_con['CRS_DEP_TIME'].tolist()
                    for i in range(len(time_in)):
                        for j in range(len(time_con)):
                            # if this is a viable connection, add to list
                            if data_con.iloc[j]['CRS_DEP_TIME'] > data_in.iloc[i]['CRS_ARR_TIME'] + delta and data_con.iloc[j]['CRS_DEP_TIME'] <= (data_in.iloc[i][ 'CRS_ARR_TIME'] + max_connection_time):
                                connection.append({'first_flight': data_in.iloc[i]['FLIGHT_NUMBER'],
                                                    'second_flight': data_con.iloc[j]['FLIGHT_NUMBER'], \
                                                    'departure_time': data_in.iloc[i]['CRS_DEP_TIME'], \
                                                    'travel_time': data_con.iloc[j]['CRS_ELAPSED_TIME']+data_in.iloc[i]['CRS_ELAPSED_TIME']+data_con.iloc[j]['CRS_DEP_TIME']-data_in.iloc[i]['CRS_ARR_TIME'], 'market': od_pair, 'connection_airport': ap,'market_obj':market_obj})
                                serve_market.append(market_obj)

        iti_list=leg_1+connection
        un_ordered_list=list(set(serve_market))
        un_ordered_list.sort()
        self.serve_market=un_ordered_list
        return iti_list
        
    def sequence_departure_generator(self): #input 'which market' and get dictionary .market_time={}and market_travel_time={}
        for market_obj in self.serve_market:
            itinerary_obj=[]
            for iti in self.itinerary:
                if iti.market_obj== market_obj:
                    itinerary_obj.append(iti)
            self.market_itinerary.update({market_obj:itinerary_obj})
            self.market_itinerary_dtime.update({market_obj:[itinerary.departure_time for itinerary in itinerary_obj]})
            self.market_itinerary_ttime.update({market_obj:[itinerary.travel_time for itinerary in itinerary_obj]})
    def get_vector(self,initial_price,initial_pax): #gengrate player's decision vector(price and pax) ,input a number, get market_vector{}
        #self.market_vector={ market_obj:np.array([initial_price]*len(value)*len(market_obj.passenger_types)+[initial_pax]*len(value)*len(market_obj.passenger_types)) for market_obj ,value in self.market_itinerary.items()}
        price_list=[]
        pax_list=[]
        index=0
        dict_record={}
        for iti_obj in self.itinerary:
            for i in iti_obj.market_obj.passenger_types_list:
                price_list.append(initial_price)
                pax_list.append(initial_pax)
                dict_record.update({(iti_obj,i):index})
                index +=1
        #self.strategy_vector_size=len(np.array(price_list+pax_list))
        self.itin2price_ind=dict_record
        self.initial_vector=np.array(price_list+pax_list)
        self.itin2pax_ind={keys:value+int(len(self.initial_vector)/2) for keys,value in self.itin2price_ind.items()}

    def flight_itinerary_record(self): #redord carrier uesed by itinerarys
        for flight in self.flights:
            index_list=[]
            itinerary_obj=[]
            for i, itinerary in enumerate(self.itinerary):
                if itinerary.first_flight==flight.flight_index or itinerary.second_flight==flight.flight_index:
                    index_list.append(i)
                    #reed added
                    itinerary_obj.append(itinerary)
            flight.itineraries=itinerary_obj
            self.flight_itinerary_dic.update({flight:index_list})

    def payoff_optimize(self,initial_vector):

        return -np.sum(initial_vector[:int(len(initial_vector) / 2)] * initial_vector[int(len(initial_vector) / 2):])
    
    def attra(self,initial_vector, iti_obj,pt, beta, strategies_dict, ):
        market_obj=iti_obj.market_obj
        market_obj.total_utility={}
        # mat_index=[(pax_index,iti_index) for pax_index in range(len(market_obj.passenger_types_list))
        #            for iti_index in range (len( self.market_itinerary[market_obj]))]
        iti_index=self.market_itinerary[market_obj].index(iti_obj)
        pt_index=market_obj.passenger_types_list.index(pt)
        for player_obj in market_obj.players:
                mat=market_obj.none_price_mat[player_obj]
                itinerary_obj_list = player_obj.market_itinerary[market_obj]
                index = []
                for itinerary_obj in itinerary_obj_list:
                    for type in market_obj.passenger_types_list:
                        index.append(player_obj.itin2price_ind[(itinerary_obj, type)])
                if self.player_index == player_obj.player_index:
                    price_vec = initial_vector[index]
                else:
                    price_vec = strategies_dict[player_obj.player_index][index]
                price_mat = np.exp(beta * price_vec.reshape(-1, len(market_obj.passenger_types_list)).T)
                utility_total = price_mat * mat
                market_obj.total_utility[player_obj]=utility_total #record in market obj #add both player utility_mat in same market
            #一个市场跑完两个player的total utility.
            # 先把两个 mat 进行拼接
        utility_mat= np.hstack((market_obj.total_utility[market_obj.players[0]], market_obj.total_utility[market_obj.players[1]]))
        market_obj.numerate_mat = np.dot(utility_mat, np.ones((utility_mat.shape[1], 1)))#横向求和
        p = market_obj.total_utility[self]/(market_obj.numerate_mat + market_obj.n_value)
        pax_number_mat = p * np.array([list(market_obj.market_demand_dic.values())]).T
        return pax_number_mat[pt_index,iti_index] - initial_vector[self.itin2pax_ind[(iti_obj, pt)]]
    def capacity(self,initial_vector,flight_obj):
        itinerary_obj_list=flight_obj.itineraries
        iti_index = []
        for itinerary_obj in itinerary_obj_list:
            pax_type_list=itinerary_obj.market_obj.passenger_types_list
            for pax_type in pax_type_list:
                iti_index.append(self.itin2pax_ind[(itinerary_obj,pax_type)])
        
        return flight_obj.capacity - np.sum(initial_vector[iti_index])
    
    def plot_con(self,con,p,index=None):
        if not index:
            x=np.array(3*[p]*len(self.itinerary) + 3*[0]*len(self.itinerary))
            return con['fun'](x,con['args'][0],con['args'][1],con['args'][2],con['args'][3],con['args'][4])

    def con(self,strateges_dic, beta ):
        cons=[]
        for flight_obj in self.flights:
            jac = grad(self.capacity)
            #hess=hessian(self.capacity)
            cons.append({'type': 'ineq', 'fun': self.capacity,'jac':jac, 'args': (flight_obj,)})

        for iti_obj in self.itinerary:
            for pt in iti_obj.market_obj.passenger_types_list:
                jac = grad(self.attra)
                cons.append({'type': 'ineq', 'fun': self.attra,'jac':jac,'args':(iti_obj,pt,beta, strateges_dic)})

        return cons

class Passenger(): # not finish
    def __init__(self,type_name,proportions,peak_time):
        self.type_name=type_name
        self.proportions=proportions
        self.peak_time=peak_time

    def __repr__( self ):
        return 'type_name :{},proportions{},peak_time:{}, '.format(self.type_name,self.proportions,self.peak_time)
    @classmethod
    def passengenr_instance_gen(cls,tup_list):  #这里cls 对比self ，self是对象本身，cls是类本身
        pax_instance_list=[]
        for i in range(len(tup_list[0])):
            pax = cls(tup_list[0][i], tup_list[1][i], 5 + 3 * i)
            pax_instance_list.append(pax)
        return  pax_instance_list

# class PassengerType：
#       def

class Market(object):
 # A Market contains a basic info in progress
    def __init__( self, market_index,od_pair,demand,distance,):
            self.market_index = market_index
            self.od_pair = od_pair
            self.demand=demand
            self.distance=distance
            self.n_value=None
            self.none_price_mat={} #{player:mat}
            self.total_utility={}
            self.passenger_types_list=None#['oneway', 'outbound', 'inbound']
            self.market_demand_dic={}
            self.pax_obj_list=[]
            self.players=[] #players' obj operate in this market

    def __repr__(self):
        return 'market_index:{},od_pair{},demand:{}, '.format(self.market_index,
                                                              self.od_pair,self.demand)
    @classmethod
    def market_instance_gen(cls,dict_demand ):
        instance_list = []
        i=1
        for od, demand in dict_demand.items():
            market = cls(i,od,demand[0],demand[1])
            i+=1
            instance_list.append(market)
        return instance_list

    def add_passenger( self,):
        for passenger_obj in self.pax_obj_list:
            self.market_demand_dic.update({passenger_obj.type_name:passenger_obj.proportions*self.demand})

        self.passenger_types_list=list(self.market_demand_dic.keys())


    def add_pax_obj(self,type_name,proportions,peak_time):
        self.pax_obj_list.append(Passenger(type_name,proportions,peak_time))

    def __lt__(self,other):

        return self.market_index<other.market_index


class Itinerary(object):
    """info of Itinerary"""
    def __init__( self,market, departure_time,travel_time, first_flight,second_flight, market_obj,connection_ap ):
        self.market_index = market
        self.departure_time = departure_time
        self.travel_time = travel_time
        self.first_flight = first_flight
        self.second_flight=second_flight
        self.market_obj=market_obj
        self.connection_ap=connection_ap
    def __repr__(self):
        return 'market:{},departure_time{},travel_time:{} ,connection_ap:{}'.format(self.market_index,
                                                                                               self.departure_time,self.travel_time,self.connection_ap )
    def __lt__(self,other):

        return self.departure_time<=other.departure_time

    @classmethod
    def itinerary_instance_gen( cls,itinerary_info_list ):
        instance_list = []
        for info_dict in itinerary_info_list:
            itinerary_obj=cls(info_dict['market'],info_dict['departure_time'],info_dict['travel_time'],info_dict['first_flight'],info_dict['second_flight'],info_dict['market_obj'],info_dict['connection_airport'])
            instance_list.append(itinerary_obj)
        return instance_list

class Flights(object):
    # capacity of each carrier
    def __init__(self,flight_index,segment,capacity,carrier,CRS_DEP_TIME):
        self.flight_index=flight_index
        self.capacity=capacity
        self.itineraries = []
        self.carrier = carrier
        self.depart_time=CRS_DEP_TIME
        self.segment=segment
    def __repr__( self ):
        return 'flight_index:{},capacity:{},carrier:{},depart_time:{} ,itineraries:{}'.format(self.flight_index,self.capacity,
        self.carrier,self.depart_time,self.itineraries)
    def __lt__( self, other ):
        return self.flight_index <= other.flight_index
    @classmethod
    def flight_instance_gen( cls, df,name ):
        instance_list = []
        for i in range(len(df)):
            a = [df.iloc[i]['FLIGHT_NUMBER'],(df.iloc[i]['ORIGIN'],df.iloc[i]['DEST']),df.iloc[i]['Capacity'],name,
                 df.iloc[i]['CRS_DEP_TIME']]
            f = cls(*a)
            instance_list.append(f)
        return instance_list

class Passenger_utility_mat_new(object):
    def compute(self,market_obj_list,player_obj_list,elapsed_time_para,connection_beta,d_time_counted):
        self.market_nonprice = {}
        for market_obj in market_obj_list:
            utility_nonprice = {}
            for player in market_obj.players:
            #if market_obj in player.market_itinerary.keys():
                iti_obj_list = player.market_itinerary[market_obj]
                d_time = np.array([itinerary.departure_time for itinerary in iti_obj_list])/60
                #test=np.array([4,5,6,7,8,9,10,11,12])
                t_time = np.array([itinerary.travel_time for itinerary in iti_obj_list])
                connections=np.array([0 if itinerary.connection_ap == "None" else 1 for itinerary in iti_obj_list])
                utility_mat=np.zeros((len(market_obj.pax_obj_list),len(d_time)))
                if d_time_counted==True:
                    for i in range(utility_mat.shape[0]):
                        #for j in range(utility_mat.shape[1]):
                        utility_mat[i]=norm(market_obj.pax_obj_list[i].peak_time,3).pdf(d_time)
                #utility_mat=utility_mat*np.array([self.co_eff_list]).T
                utility_elasped_time = elapsed_time_para * t_time

                utility_nonprice_value = np.exp(
                    utility_elasped_time + utility_mat + connections * connection_beta)
                utility_nonprice.update({player: utility_nonprice_value})
            market_obj.none_price_mat = utility_nonprice
            self.market_nonprice.update({market_obj: utility_nonprice})

#para = read_departure_para(excelfile)
class Passenger_utility_mat(object):
    def __init__(self,TODS_number,daytime_number,para_data):
        self.para_data=para_data
        self.TODS_number=TODS_number
        self.daytime_number=daytime_number
        self.para_mat='none'
    def compute(self,market_obj_list,player_obj_list,elapsed_time_para):
        df = self.para_data[list(self.para_data.keys())[self.TODS_number + 1]]
        self.para_mat = np.array(df.iloc[self.daytime_number - 1:self.daytime_number + 2])
        self.market_nonprice = {}
        for market_obj in market_obj_list:
            utility_nonprice = {}
            for player in player_obj_list:
                iti_obj_list=player.market_itinerary[market_obj]
                d_time=np.array([itinerary.departure_time for itinerary in iti_obj_list])
                t_time = np.array([itinerary.travel_time for itinerary in iti_obj_list])
                sc_fun = np.array([np.sin((2 * d_time * math.pi) / 1440), np.sin((4 * d_time * math.pi) / 1440),
                                   np.sin((6 * d_time * math.pi) / 1440),
                                   np.cos((2 * d_time * math.pi) / 1440), np.cos((4 * d_time * math.pi) / 1440),
                                   np.cos((6 * d_time* math.pi) / 1440)])
                departure_passenger_mat = np.dot(self.para_mat, sc_fun)
                utility_elasped_time = elapsed_time_para * t_time
                utility_nonprice_value=np.exp(utility_elasped_time + departure_passenger_mat + market_obj.connections * -2.66)
                utility_nonprice.update({player:utility_nonprice_value})

            market_obj.none_price_mat=utility_nonprice
            self.market_nonprice.update({market_obj:utility_nonprice})


class Game(object):
    def __init__(self,market_objects,player_objects ):
        self.players = player_objects  # list of players
        self.markets = market_objects  # list of markets

    def create_initial_strategies_object( self, initial_price ,initial_passengers=0,):
        Strategies = {}
        Profits={}
        # create a dictionary of initial strategies for each player
        for i, player in enumerate(self.players):
            player.get_vector(initial_price,initial_passengers)
            Strategies[player.player_index] = player.initial_vector
            Profits[player.player_index]='none'
        return Strategies,Profits

    # function to check for convergence of best response
    def is_converged( self, Strategies, Strategies_previous, tolerance ):
        diff = False
        r=[]
        for player in self.players:
            a=abs(Strategies[player.player_index][0:int(len(Strategies[player.player_index])/2)] -Strategies_previous[player.player_index][0:int(len(Strategies_previous[player.player_index])/2)])
            r.extend(a)
            #diff+= a
        b=nu.where(np.array(r)>tolerance)[0]
        if not any(b):
            diff=True

        return diff
        
    def optimization_callback(self,Xi):        
         
        print(Xi)

    def  solve_game_best_response( self, initial_price, initial_pax, tolerance, beta ):
        # create a dummy strategies placeholder so best response doesnt converge immediately
        Strategies_previous,Profits = self.create_initial_strategies_object(initial_price - tolerance-1,initial_pax - tolerance)
        Strategies,Profits= self.create_initial_strategies_object(initial_price,initial_pax)
        # keep list of strategies for each round
        strategies_history = []
        strategies_history.append(Strategies_previous.copy())
        optimization_results = []
        # best response

        while not self.is_converged(Strategies, Strategies_previous, tolerance):
            Strategies_previous = Strategies.copy()
            for i, player in enumerate(self.players):
                cons = player.con( Strategies, beta)
                bnds = [(0,None)]*len( Strategies[player.player_index])
                jac = grad(player.payoff_optimize)
                res_start_time=time.perf_counter()
                print('minimize_ipopt is running')
                res = minimize(player.payoff_optimize,Strategies[player.player_index],method='SLSQP',
                               jac=jac,constraints=cons,bounds=bnds,options={'ftol':100,'eps':1e-08,'iprint': 1,'disp':True})
                # res = minimize_ipopt(player.payoff_optimize, Strategies[player.player_index],constraints=cons, bounds=bnds,tol=1e8,options={'disp': 1})
                #print(np.sum(player.attra_probability_list[0],1))
                res_end_time=time.perf_counter()
                print("player_name is:",player.name)
                print('res cost is :', res_end_time - res_start_time)
                #res = minimize(player.payoff_optimize, Strategies[player.player_index], constraints=cons, bounds=bnds,callback=self.optimization_callback,options={'ftol':1,'eps':1e-6,'disp':True)
                Strategies[player.player_index] = res.x
                Profits[player.player_index] = -res.fun
                optimization_results.append(res)
                #break
            # add to player history
            strategies_history.append(Strategies.copy())
        for  player in self.players:
            player.initial_vector=Strategies[player.player_index]
            player.itin2price={keys:player.initial_vector[index] for keys,index in player.itin2price_ind.items()}
            player.itin2pax = {keys: player.initial_vector[index] for keys, index in player.itin2pax_ind.items()}
            player.profit=Profits[player.player_index]

        return Strategies, strategies_history, optimization_results,Profits


# class Solution(Game):
#     '''
#     Class Solution is a solution that contains a timetable , player's obj (player_obj contain profit and flight informations)
#     Attributions:
#     market_objects: list of market_obj
#     player_objects:list of player_obj
#     dataframe:read from excel ,a time table
#     hub_name: define which airport is connecting hub
#     '''
#     def __init__( self, market_objects, player_objects, dataframe,hub_name):
#         super(Solution,self).__init__(market_objects, player_objects)
#         self.df=dataframe
#         self.hub=hub_name
#         #self.df_columns_list=self.df.columns.values.tolist()
#         self.player_list = list(set(self.df['UNIQUE_CARRIER']))
#         self.number_of_player=len(self.player_list)
#         self.flight_number_total=self.df['UNIQUE_CARRIER'].shape[0]
#         self.player_flights_number={name:self.df.loc[self.df['UNIQUE_CARRIER']==name].shape[0] for name in self.player_list }
#         #self.player_data=None
#         self.flight_result_info=None
#         self.chosen_flight_index=None #flight_index
#         self.chosen_flight_relevant_itinerary=None #list  [itinerary_index,itinerary_index]
#     def itinerary_generator_new(self,delta,max_connection_time):
#         '''
#         generate all itinerarise and store them into a dictionary
#         :param delta: maxima connection time
#         :return: a dictionary like{player: list of itinerary_obj}
#         '''
#         def itinerary_connection_generator(locations_in,locations_con,data_in,data_con,delta,max_connection_time ):
#             # get connections
#             connections = []  # list of connections - input first flight, get second flight
#             leg_1 = []
#             leg_2 = []
#             for i in  range(len(locations_in)):
#                 leg_1.append({'first_flight': data_in.iloc[i]['FLIGHT_NUMBER'], 'second_flight': 'none',
#                               'departure_time': data_in.iloc[i]['CRS_DEP_TIME'],  # '2nd_departure_time':'none',\
#                               'travel_time': data_in.iloc[i]['CRS_ELAPSED_TIME'], 'market': 0})
#                 for j in range(len(locations_con)):
#                     # if this is a viable connection, add to list
#                     if data_con.iloc[j]['CRS_DEP_TIME'] > data_in.iloc[i]['CRS_ARR_TIME'] + delta and data_con.iloc[j]['CRS_DEP_TIME'] <= data_in.iloc[i]['CRS_ARR_TIME'] + max_connection_time:
#                         connections.append({'first_flight': data_in.iloc[i]['FLIGHT_NUMBER'],
#                                             'second_flight': data_con.iloc[j]['FLIGHT_NUMBER'], \
#                                             'departure_time': data_in.iloc[i]['CRS_DEP_TIME'], \
#                                             # '2nd_departure_time':data_con.iloc[j]['CRS_DEP_TIME'],\
#                                             'travel_time': data_con.iloc[j]['CRS_ARR_TIME'] - data_in.iloc[i][
#                                                 'CRS_DEP_TIME'], 'market': 2})
#                         #break  # stop because all other connections will be longer (since data is sorted)
#             for i in range(len(locations_con)):
#                 leg_2.append({'first_flight': data_con.iloc[i]['FLIGHT_NUMBER'], 'second_flight': 'none', \
#                               'departure_time': data_con.iloc[i]['CRS_DEP_TIME'], \
#                               'travel_time': data_con.iloc[i]['CRS_ELAPSED_TIME'], \
#                               'market': 1})
#             iti_list = leg_1 + leg_2 + connections
#             return iti_list
#         self.player_data={name:self.df[(self.df['UNIQUE_CARRIER'] == name)].sort_values(by='CRS_DEP_TIME') for name in self.player_list }
#         iti_dict = {}
#         for player in self.player_list:
#             data_in=self.player_data[player][self.player_data[player]['DEST'] == self.hub].sort_values(by='CRS_DEP_TIME')
#             data_con = self.player_data[player][self.player_data[player]['ORIGIN'] == self.hub].sort_values(by='CRS_DEP_TIME')
#             locations_in = data_in['CRS_DEP_TIME'].tolist()
#             locations_con = data_con['CRS_DEP_TIME'].tolist()
#             itinerary_total = itinerary_connection_generator(locations_in, locations_con, data_in, data_con,delta,max_connection_time)
#             iti_dict[player] = itinerary_total
#         self.itinerary_info = iti_dict
#         itinerary_record = {}
#         for player, itinerary_list in self.itinerary_info.items():
#             itinerary = []
#             for info_dict in itinerary_list:
#                 info_dict.update({'market_obj': self.markets[info_dict['market']]})
#                 itinerary_obj = Itinerary(**info_dict)
#                 itinerary.append(itinerary_obj)
#             itinerary_record.update({player: itinerary})
#         self.itinerary_record_player=itinerary_record
#
#
#     def gen_flightobj_record(self):
#         flight_obj_record = {}
#         for player in self.player_list:
#             flight=self.player_data[player][['FLIGHT_NUMBER', 'UNIQUE_CARRIER', 'Capacity','CRS_DEP_TIME']]
#             flight_obj_list = []
#             for i in range(len(flight)):
#                 a = [flight.iloc[i]['FLIGHT_NUMBER'], flight.iloc[i]['Capacity'], 'NW',flight.iloc[i]['CRS_DEP_TIME']]
#                 f = Flights(*a)
#                 flight_obj_list.append(f)
#             flight_obj_record.update({player:flight_obj_list})
#         self.player_flight_dic= flight_obj_record
#
#     def player_add_itinerary_flight(self):
#         '''
#         :return: add itinerary_obj and flight_obj to each player_obj
#         '''
#         for obj in self.players:
#             obj.add_itinerary(*self.itinerary_record_player[obj.name])
#             obj.add_flights(*self.player_flight_dic[obj.name])
#     def player_setting(self):
#         for obj in self.players:
#             obj.serve_market = self.markets
#             obj.sequence_departure_generator()
#             obj.flight_itinerary_record()
#
#     def market_nonprice_mat(self,co_eff):
#         '''
#         :param co_eff:
#         :return:market_obj get non_price utility mat
#         '''
#         D1 = Passenger_utility_mat_new(co_eff)
#
#         D1.compute(self.markets,self.players,-0.00387 )
#
#     def solve_game_best_response(self, initial_price, initial_pax, tolerance, beta ):
#         '''
#         compute the equilibrium, get the profit etc
#         :param initial_price:
#         :param initial_pax:
#         :param tolerance:
#         :param beta:
#         :return:
#         '''
#         game_start_time=time.perf_counter()
#         Strategies, strategies_history, optimization_results, Profits=Game.solve_game_best_response(self,initial_price, initial_pax, tolerance, beta)
#         game_end_time = time.perf_counter()
#         print('one game cost is :', game_end_time-game_start_time)
#         self.profit={player.name:Profits[player.player_index ] for player in self.players}
#     def get_min_load_factor_flight(self,player_name,jump=False):
#          flight_pax = []
#          for player_obj in self.players:
#              for flight_obj in player_obj.flights:
#                  li = []
#                  d_time = []
#                  for itinerary_obj in flight_obj.itineraries:
#                      pax_iti = sum([v for k, v in player_obj.itin2pax.items() if k[0] == itinerary_obj])
#                      li.append(pax_iti)
#                      d_time.append(itinerary_obj.departure_time)
#                  pax = sum(li)
#                  f_dic = {'flight': flight_obj.flight_index, 'player': player_obj.name, 'capacity': flight_obj.capacity,
#                           'flight_pax': pax, \
#                           'pax_per_iti': tuple(li), 'depart_time': d_time}
#                  flight_pax.append(f_dic)
#          df1 = pd.DataFrame(flight_pax)
#          df1.eval('load_factor=flight_pax/capacity', inplace=True)
#          self.flight_result_info=df1
#          df1_player=df1.loc[df1['player'] == player_name].sort_values('load_factor')
#          if not jump:
#             self.chosen_flight_index=df1_player.iloc[0,:]['flight']# {player_obj:flight_index}
#          else:
#              self.chosen_flight_index = df1_player.iloc[1,:]['flight']
#          #self.chosen_flight_relvant_itinerary=[player.flight_itinerary_dic[flight] for flight in player.flight_itinerary_dic.keys() if flight.flight_index == self.chosen_flight_index[player]][0]
#
#     def get_connecting_flight(self):
#         for player_obj in self.players:
#             for flight_obj in player_obj.flights:
#                 if flight_obj.flight_index == self.chosen_flight_index:
#                    self.connected_flights = [(iti.first_flight,iti.second_flight)for iti in flight_obj.itineraries]

'''      
    def get_min_load_factor_flight(self,player_obj):
        flight_pax = []
        for player_obj in self.player_list:
            for flight_obj in p，layer_obj.flights:
                li = []
                d_time = []
                for itinerary_obj in flight_obj.itineraries:
                    pax_iti = sum([v for k, v in player_obj.itin2pax.items() if k[0] == itinerary_obj])
                    li.append(pax_iti)
                    d_time.append(itinerary_obj.departure_time)
                pax = sum(li)
                f_dic = {'flight': flight_obj.flight_index, 'player': player_obj.name, 'capacity': flight_obj.capacity,
                         'flight_pax': pax,'pax_per_iti': tuple(li), 'depart_time': d_time}
                flight_pax.append(f_dic)
        df1 = pd.DataFrame(flight_pax)
        df1.eval('load_factor=flight_pax/capacity', inplace=True)
'''


