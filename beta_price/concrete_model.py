# -*- coding: utf-8 -*-
"""
Created on 2020/7/16  15:47 

# Author： Jinyu
"""
# -*- coding: utf-8 -*-
from pyomo.pysp.ef import solve_ef

"""
Created on 2020/6/18  14:34 

# Author： Jinyu
"""
from pyomo.environ import *
from pyomo.core.base import NonNegativeReals,ConcreteModel,Set,Param,PositiveIntegers,RangeSet,Var,Binary,Constraint,Objective,maximize,Expression
from pyomo.core import value as value_s
import numpy as np
import pandas as pd
import pyomo.pysp.util.rapper

def set_model(market_instances_list, airport_list, market_price,market_data, iti_dict, fleet_data, passeger_type_dict, attr_value,
               marketpt_attr_sum, solver ):
    """
    很多输入变量都是字典，就是为了在模型中生成子集的时候有筛选功能 即 if dict【m】 in dict 
    :param airport_list: 
    :param market_data: 
    :param segment_travel_time: 
    :param iti_dict:itinerary number key ,info value :{0: {'market': "M('SEA', 'LAX')", 'non_stop': ('SEA', 'LAX'), ('SEA', 'LAX'): 1, 'legs': [('SEA', 'LAX')]}
    :param fleet_data: 
    :param passeger_type_dict: passenger_type as key , and it's proportions
    :param attr_value: 
    :param marketpt_attr_sum: 
    :param solver: 
    :return: time_table, q_variable for each itinerary, 
    """
    # market_iti = {} #market:itinerary list
    # for m in market_data.keys():
    #     l = [i for i, v in iti_dict.items() if v['market'] == m]
    #     market_iti[m]=l
    market_iti = {}  # market:itinerary list
    for m in market_instances_list:
        l = [i for i, v in iti_dict.items() if v['market'] == 'M' + str(m.od_pair)]
        market_iti[ 'M' + str(m.od_pair)] = l
    model = ConcreteModel()
    #market = list(market_data.keys())
    market=list(market_iti.keys())
    # market_segment={m:[(market_data[m][4],market_data[m][5])] for m in market_data.keys()}
    model.M = Set(initialize=market, doc='Player Market_obj')
    model.Mi = Set(list(market_iti.keys()), initialize=market_iti, doc='Player Market_obj')
    model.AP = Set(initialize=airport_list)
    model.Segment = Set(initialize=((i, j) for i in model.AP for j in model.AP if i != j))

    # create time spot [1-72] and it's time
    a = np.linspace(1, int((1440 - 360)/ 15), int((1440- 360) /15))
    time_list = np.arange(360, 1440, 15)
    time_dict = dict(zip(a, time_list))
    # reverse_time_dict = {v: k for k, v in time_dict.items()}
    model.T = Set(initialize=a, ordered=True, doc='Time period from 300 min to 1440 ,step :15min,number:73')
    model.I = Set(initialize=iti_dict.keys(),doc='itinerary _index,size 4334')
    model.Im = Set(initialize=((m, i) for m in model.M for i in market_iti[m]),doc='tuple(m,i),size 4334') # 筛选出只在m的itinerary 的 号码
    model.PT = Set(initialize=passeger_type_dict.keys())
    model.F = Set(initialize=fleet_data.keys())
    d = {}  #create a dict as which OD and time get all itinerary_index in it
    for ap1, ap2 in model.Segment:
        for t in model.T:
            v = []
            for i, va in iti_dict.items():
                if (ap1, ap2) in va['legs'] and t == va[(ap1, ap2)]:
                    v.append(i)
            d[(ap1, ap2, t)] = v
    model.St = Set(list(d.keys()), initialize=d)  # index as (ap1,ap2,time) get [itinerary index list]

    def _filter3( model, i, m, ap1, ap2, t ):

        return i in model.Mi[m].value and i in model.St[(ap1, ap2, t)].value
    # def Parameters
    # demand_pt = {} # get demand of pt in the market by total demand times its proportion
    # print("passenger_type_proportion:", passeger_type_dict)
    # for m, va in market_data.items():
    #     for ty,rato in passeger_type_dict.items():
    #         demand_pt.update({(m, ty): va[0] * rato})  # market demand times proportions
    demand_pt={}
    for mar in market_instances_list:
        for pax_obj in mar.pax_obj_list:
            demand_pt[( 'M' + str(mar.od_pair), pax_obj.type_name)]=pax_obj.proportions * mar.demand
    model.Dem = Param(model.M, model.PT, initialize=demand_pt, doc='Market demand for each type of passenger')

    market_price={'M'+str(key):v for key ,v in market_price.items()}
    price_dict = {}#这个价格应该是根据每个itinerary设定一个价格 ，这点以后需要改
    for i in model.I:
        rato = np.linspace(1.3, 0.7, len(passeger_type_dict))
        for index, ty in enumerate(passeger_type_dict):
            if iti_dict[i]['non_stop']:
                price_dict[(i, ty)]=market_price[iti_dict[i]['market']]* rato[index]# market_data[m][2] is price
            else:
                price_dict[(i, ty)] = market_price[iti_dict[i]['market']] * rato[index]  # market_data[m][2] is price
    model.p = Param(model.I, model.PT, initialize=price_dict)
    model.Avail = Param(model.F, initialize={fleet: value[0] for fleet, value in fleet_data.items()})
    model.Cap = Param(model.F, initialize={fleet: value[1] for fleet, value in fleet_data.items()})

    model.distance = Param(model.Segment,
                           initialize={(m.od_pair[0],m.od_pair[1]):m.distance for m in market_instances_list })

    def ope_cost( model, ap1, ap2, f ):
        if model.distance[ap1, ap2] <= 3106:
            return (1.6 * model.distance[ap1, ap2] + 722) * (model.Cap[f] + 104) * 0.019
        else:
            return (1.6 * model.distance[ap1, ap2] + 2200) * (model.Cap[f] + 211) * 0.0115

    model.Ope = Param(model.Segment, model.F, initialize=ope_cost, doc='cost')
    freq = {(market_data[m][4], market_data[m][5]): market_data[m][3] for m in market_data.keys()}

    model.Freq = Param(model.Segment, initialize=freq)
    model.A = Param(model.I, model.PT, initialize=attr_value)

    model.Am = Param(model.M, model.PT, initialize=marketpt_attr_sum)
    # Step 2: Define decision variables
    model.x = Var(model.Segment, model.F, model.T, within=Binary)
    model.y_1 = Var(model.F, model.AP, model.T, within=PositiveIntegers)
    model.y_2 = Var(model.F, model.AP, model.T, within=PositiveIntegers)
    model.q = Var(model.I, model.PT, within=NonNegativeReals)
    model.non_q = Var(model.M, model.PT, within=NonNegativeReals)  # number of pax that choose others airine and no fly.

    # Step 3: Define Objective
    def obj_rule(model):
        return sum(model.q[i, pt] * model.p[i, pt] for i in model.I for pt in model.PT) - sum(
            model.Ope[s, f] * model.x[s, f, t] for s in model.Segment for f in model.F for t in model.T)

    model.obj = Objective(rule=obj_rule, sense=maximize)
    def obj_cost(model):
        return sum(model.Ope[s, f] * model.x[s, f, t] for s in model.Segment for f in model.F for t in model.T)

    model.obj_cost = Expression(rule=obj_cost)
    def obj_revenue(model):
        return sum(model.q[i, pt] * model.p[i, pt] for i in model.I for pt in model.PT)

    model.obj_revenue = Expression(rule=obj_revenue)
    # add constraint
    # Aircraft count:
    def aircraft_con( model, f ):
        return sum(model.y_1[f, ap, model.T[1]] for ap in model.AP) <= model.Avail[f]

    model.count = Constraint(model.F, rule=aircraft_con)

    # flow balance cons
    def flow_balance_1( model, f, ap ):
        return model.y_1[f, ap, model.T[1]] == model.y_2[f, ap, model.T[-1]]

    model.con_fb_1 = Constraint(model.F, model.AP, rule=flow_balance_1)

    def flow_balance_2( model, f, ap, t ):
        # if t == model.T[-1]:
        #     return Constraint.Skip
        # else:
        return model.y_1[f, ap, t + 1] == model.y_2[f, ap, t]
    def filter2(model,t):
        return t != model.T[-1]
    model.Tm = Set(initialize=(i for i in model.T if i != model.T[-1]))
    #model.con_fb_2 = Constraint(model.F, model.AP, model.T, rule=flow_balance_2)
    model.con_fb_2 = Constraint(model.F, model.AP,model.Tm, rule=flow_balance_2)
    # time_zone_dict={('ANC', 'PDX'): 60, ('SEA', 'PDX'): 0, ('SEA', 'ANC'): -60, ('ANC', 'LAX'): 60, ('PDX', 'SEA'): 0, ('LAX', 'PDX'): 0,
    #                 ('LAX', 'SEA'): 0, ('PDX', 'ANC'): -60, ('SEA', 'LAX'): 0, ('ANC', 'SEA'): 60, ('LAX', 'ANC'): -60, ('PDX', 'LAX'): 0}


    Travel_time = {m.od_pair:m.travel_time for m in market_instances_list}
    time_zone_dict = {m.od_pair: m.mistiming for m in market_instances_list}
    def flow_balance_3( model, f, ap, t ):
        def D( s, t, turnaround=30 ):
            arrival_time = time_dict[t]
            dep_time = arrival_time - (Travel_time[s] + turnaround)-time_zone_dict[s]
            if dep_time >= 360:
                t0 = ((dep_time - 360) // 15) + 1
            else:
                t0 = 72 - (abs(360 - dep_time) // 15)
            return t0

        return model.y_1[f, ap, t] + sum(model.x[s, f, D(s, t)] for s in model.Segment if s[1] == ap) == model.y_2[f, ap, t] + sum(
            model.x[s, f, t] for s in model.Segment if s[0] == ap)

    model.con_fb_3 = Constraint(model.F, model.AP, model.T, rule=flow_balance_3)

    # Demand and capacity constrains:
    def attract_con( model, market, i, pt ):
        return model.Am[market, pt] * (model.q[i, pt] / model.Dem[market, pt]) <= model.A[i, pt] * (
            model.non_q[market, pt] / model.Dem[market, pt])

    model.attractiveness = Constraint(model.Im, model.PT, rule=attract_con)

    def capacity_con( model, ap1, ap2, t ):
        return sum(model.q[i, pt] for i in d[(ap1, ap2, t)] for pt in model.PT) <= sum(
            model.Cap[f] * model.x[ap1, ap2, f, t] for f in model.F)

    model.con_d1 = Constraint(model.Segment, model.T, rule=capacity_con)

    def demand_market_con( model, market, pt ):
        return sum(model.q[i, pt] for i in model.I if iti_dict[i]['market'] == market) + model.non_q[market, pt] == \
               model.Dem[market, pt]

    model.con_d3 = Constraint(model.M, model.PT, rule=demand_market_con)

    # Itinerary selection constraints:
    model.AC = Set(initialize=model.I * model.M * model.Segment * model.T, filter=_filter3)

    def iti_selection( model, i, m, ap1, ap2, t, pt ):
        # if i in market_iti[m] and i in d[(ap1, ap2, t)]:
        return sum(model.x[ap1, ap2, f, t] for f in model.F) >= model.q[i, pt] / model.Dem[m, pt]

    model.con_iti_selection = Constraint(model.AC, model.PT, rule=iti_selection)
    # Restrictions on fight leg variables:
    def flight_leg_con( model, ap1, ap2, t ):
        return sum(model.x[ap1, ap2, f, t] for f in model.F) <= 1

    model.con_leg_1 = Constraint(model.Segment, model.T, rule=flight_leg_con)

    def freq_con( model, ap1, ap2 ):
        return sum(model.x[ap1, ap2, f, t] for t in model.T for f in model.F) == model.Freq[ap1, ap2]

    model.con_let_2 = Constraint(model.Segment, rule=freq_con)
    print("____" * 5)
    # for con in model.component_map(Constraint).itervalues():
    #     con.pprint()

    SOLVER_NAME = solver
    TIME_LIMIT = 60*60*2
    results = SolverFactory(SOLVER_NAME)

    if SOLVER_NAME == 'cplex':
        results.options['timelimit'] = TIME_LIMIT
    elif SOLVER_NAME == 'glpk':
        results.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':
        results.options['TimeLimit'] = TIME_LIMIT

    com = results.solve(model,tee=True)
    com.write()


    #absgap = com.solution(0).gap
    # get x results in matrix form
    df_x = pd.DataFrame(columns=list(model.Segment) ,index=model.T)
    for s in model.Segment:
        for t in model.T:
            for f in model.F:
                if model.x[s,f,t].value > 0:
                    df_x.loc[t, [s]] = f

    #df_x=df_x.reset_index()# return value  is a dataframe of new time table
    # 所有的决策变量都遍历一遍
    # for v in instance.component_objects(Var, active=True):
    #     print("Variable", v)
    #     varobject = getattr(instance, str(v))
    #     for index in varobject:
    #         print("   ", index, varobject[index].value)
    varobject = getattr(model, 'q')
    q_data = {(i,pt): varobject[(i, pt)].value for (i, pt), v in varobject.items() if  varobject[(i, pt)] !=0}
    df_q = pd.DataFrame.from_dict(q_data, orient="index", columns=["variable value"])
    varobject2 = getattr(model,'non_q')
    nonq_data = {(m,pt): varobject2[(m, pt)].value for (m, pt), v in varobject2.items() if  varobject2[(m, pt)] !=0}

    # q = list(model.q.get_values().values())
    # print('q = ', q)
    profit=model.obj()
    print('\nProfit = ', profit)
    cost=value_s(model.obj_cost())
    #revenue=value_s(model.obj_revenue())
    print('cost is:'*10,cost)
    #print('revenue is:' * 10, revenue)
    '''
    print('\nDecision Variables')
    #list_of_vars = [v.value for v in model.component_objects(ctype=Var, active=True, descend_into=True)]
    #var_names = [v.name for v in model.component_objects(ctype=Var, active=True, descend_into=True) if v.value!=0]

    # print("y=",y)
    model.obj.display()

    def pyomo_postprocess( options=None, instance=None, results=None ):
        model.x.display()

    pyomo_postprocess(None, model, results)
    for v in model.component_objects(Var, active=True):
        print("Variable component object", v)
        varobject = getattr(model, str(v))
        for index in varobject:
            if varobject[index].value != 0:
                print("   ", index, varobject[index].value)
    '''
    return df_x,q_data,profit,cost,nonq_data


def set_testl( market_instances_list,airport_list, market_data, segment_travel_time,time_zone_dict, iti_dict, fleet_data, passeger_type_dict, attr_value,
               marketpt_attr_sum, solver ):
    """
    很多输入变量都是字典，就是为了在模型中生成子集的时候有筛选功能 即 if dict【m】 in dict 
    :param airport_list: 
    :param market_data: 
    :param segment_travel_time: 
    :param iti_dict:itinerary number key ,info value :{0: {'market': "M('SEA', 'LAX')", 'non_stop': ('SEA', 'LAX'), ('SEA', 'LAX'): 1, 'legs': [('SEA', 'LAX')]}
    :param fleet_data: 
    :param passeger_type_dict: passenger_type as key , and it's proportions
    :param attr_value: 
    :param marketpt_attr_sum: 
    :param solver: 
    :return: time_table, q_variable for each itinerary, 
    """
    market_iti = {} #market:itinerary list
    for m in market_instances_list:
        l = [i for i, v in iti_dict.items() if v['market'] == 'M' + str(m.od_pair)]
        market_iti.update({m: l})
    model = ConcreteModel()
    market = list(market_data.keys())#['M' + str(m.od_pair) for m in ]
    # market_segment={m:[(market_data[m][4],market_data[m][5])] for m in market_data.keys()}
    model.M = Set(initialize=market, doc='Player Market_obj')
    model.Mi = Set(list(market_iti.keys()), initialize=market_iti, doc='Player Market_obj')
    model.AP = Set(initialize=airport_list)
    model.Segment = Set(initialize=((i, j) for i in model.AP for j in model.AP if i != j))

    # create time spot [1-72] and it's time
    a = np.linspace(1, int((1440 - 360)/ 15), int((1440- 360) /15))
    time_list = np.arange(360, 1440, 15)
    time_dict = dict(zip(a, time_list))
    # reverse_time_dict = {v: k for k, v in time_dict.items()}
    model.T = Set(initialize=a, ordered=True, doc='Time period from 300 min to 1440 ,step :15min,number:73')
    model.I = Set(initialize=iti_dict.keys(),doc='itinerary _index,size 4334')
    model.Im = Set(initialize=((m, i) for m in model.M for i in market_iti[m]),doc='tuple(m,i),size 4334') # 筛选出只在m的itinerary 的 号码
    model.PT = Set(initialize=passeger_type_dict.keys())
    model.F = Set(initialize=fleet_data.keys())
    d = {}  #create a dict as which OD and time get all itinerary_index in it
    for ap1, ap2 in model.Segment:
        for t in model.T:
            v = []
            for i, va in iti_dict.items():
                if (ap1, ap2) in va['legs'] and t == va[(ap1, ap2)]:
                    v.append(i)
            d[(ap1, ap2, t)] = v
    model.St = Set(list(d.keys()), initialize=d)  # index as (ap1,ap2,time) get [itinerary index list]

    def _filter3( model, i, m, ap1, ap2, t ):

        return i in model.Mi[m].value and i in model.St[(ap1, ap2, t)].value
    # def Parameters
    demand_pt = {} # get demand of pt in the market by total demand times its proportion
    print("passenger_type_proportion:", passeger_type_dict)
    for m, va in market_data.items():
        for ty,rato in passeger_type_dict.items():
            demand_pt.update({(m, ty): va[0] * rato})  # market demand times proportions
    model.Dem = Param(model.M, model.PT, initialize=demand_pt, doc='Market demand for each type of passenger')

    price_dict = {}
    for i in model.I:
        rato = np.linspace(1.3, 0.7, len(passeger_type_dict))
        for index, ty in enumerate(passeger_type_dict):
            if iti_dict[i]['non_stop']:
                price_dict.update(
                    {(i, ty): market_data[iti_dict[i]['market']][2] * rato[index]})  # market_data[m][2] is price
            else:
                price_dict.update({(i, ty): 0.8 * market_data[iti_dict[i]['market']][2] * rato[
                    index]})  # market_data[m][2] is price
    model.p = Param(model.I, model.PT, initialize=price_dict)
    model.Avail = Param(model.F, initialize={fleet: value[0] for fleet, value in fleet_data.items()})
    model.Cap = Param(model.F, initialize={fleet: value[1] for fleet, value in fleet_data.items()})
    model.distance = Param(model.Segment,
                           initialize={(value[-2], value[-1]): value[1] for value in market_data.values()})

    def ope_cost( model, ap1, ap2, f ):
        if model.distance[ap1, ap2] <= 3106:
            return (1.6 * model.distance[ap1, ap2] + 722) * (model.Cap[f] + 104) * 0.019
        else:
            return (1.6 * model.distance[ap1, ap2] + 2200) * (model.Cap[f] + 211) * 0.0115

    model.Ope = Param(model.Segment, model.F, initialize=ope_cost, doc='cost')
    freq = {(market_data[m][4], market_data[m][5]): market_data[m][3] for m in market_data.keys()}

    model.Freq = Param(model.Segment, initialize=freq)
    model.A = Param(model.I, model.PT, initialize=attr_value)

    model.Am = Param(model.M, model.PT, initialize=marketpt_attr_sum)
    # Step 2: Define decision variables
    model.x = Var(model.Segment, model.F, model.T, within=Binary)
    model.y_1 = Var(model.F, model.AP, model.T, within=PositiveIntegers)
    model.y_2 = Var(model.F, model.AP, model.T, within=PositiveIntegers)
    model.q = Var(model.I, model.PT, within=NonNegativeReals)
    model.non_q = Var(model.M, model.PT, within=NonNegativeReals)  # number of pax that choose others airine and no fly.

    # Step 3: Define Objective
    def obj_rule(model):
        return sum(model.q[i, pt] * model.p[i, pt] for i in model.I for pt in model.PT) - sum(
            model.Ope[s, f] * model.x[s, f, t] for s in model.Segment for f in model.F for t in model.T)

    model.obj = Objective(rule=obj_rule, sense=maximize)
    def obj_cost(model):
        return sum(model.Ope[s, f] * model.x[s, f, t] for s in model.Segment for f in model.F for t in model.T)

    model.obj_cost = Expression(rule=obj_cost)
    def obj_revenue(model):
        return sum(model.q[i, pt] * model.p[i, pt] for i in model.I for pt in model.PT)

    model.obj_revenue = Expression(rule=obj_revenue)
    # add constraint
    # Aircraft count:
    def aircraft_con( model, f ):
        return sum(model.y_1[f, ap, model.T[1]] for ap in model.AP) <= model.Avail[f]

    model.count = Constraint(model.F, rule=aircraft_con)

    # flow balance cons
    def flow_balance_1( model, f, ap ):
        return model.y_1[f, ap, model.T[1]] == model.y_2[f, ap, model.T[-1]]

    model.con_fb_1 = Constraint(model.F, model.AP, rule=flow_balance_1)

    def flow_balance_2( model, f, ap, t ):
        # if t == model.T[-1]:
        #     return Constraint.Skip
        # else:
        return model.y_1[f, ap, t + 1] == model.y_2[f, ap, t]
    def filter2(model,t):
        return t != model.T[-1]
    model.Tm = Set(initialize=(i for i in model.T if i != model.T[-1]))
    #model.con_fb_2 = Constraint(model.F, model.AP, model.T, rule=flow_balance_2)
    model.con_fb_2 = Constraint(model.F, model.AP,model.Tm, rule=flow_balance_2)
    # time_zone_dict={('ANC', 'PDX'): 60, ('SEA', 'PDX'): 0, ('SEA', 'ANC'): -60, ('ANC', 'LAX'): 60, ('PDX', 'SEA'): 0, ('LAX', 'PDX'): 0,
    #                 ('LAX', 'SEA'): 0, ('PDX', 'ANC'): -60, ('SEA', 'LAX'): 0, ('ANC', 'SEA'): 60, ('LAX', 'ANC'): -60, ('PDX', 'LAX'): 0}


    Travel_time = segment_travel_time
    def flow_balance_3( model, f, ap, t ):
        def D( s, t, turnaround=30 ):
            arrival_time = time_dict[t]
            dep_time = arrival_time - (Travel_time[s] + turnaround)-time_zone_dict[s]
            if dep_time >= 360:
                t0 = ((dep_time - 360) // 15) + 1
            else:
                t0 = 72 - (abs(360 - dep_time) // 15)
            return t0

        return model.y_1[f, ap, t] + sum(model.x[s, f, D(s, t)] for s in model.Segment if s[1] == ap) == model.y_2[f, ap, t] + sum(
            model.x[s, f, t] for s in model.Segment if s[0] == ap)

    model.con_fb_3 = Constraint(model.F, model.AP, model.T, rule=flow_balance_3)

    # Demand and capacity constrains:
    def attract_con( model, market, i, pt ):
        return model.Am[market, pt] * (model.q[i, pt] / model.Dem[market, pt]) <= model.A[i, pt] * (
        model.non_q[market, pt] / model.Dem[market, pt])

    model.attractiveness = Constraint(model.Im, model.PT, rule=attract_con)

    def capacity_con( model, ap1, ap2, t ):
        return sum(model.q[i, pt] for i in d[(ap1, ap2, t)] for pt in model.PT) <= sum(
            model.Cap[f] * model.x[ap1, ap2, f, t] for f in model.F)

    model.con_d1 = Constraint(model.Segment, model.T, rule=capacity_con)

    def demand_market_con( model, market, pt ):
        return sum(model.q[i, pt] for i in model.I if iti_dict[i]['market'] == market) + model.non_q[market, pt] == \
               model.Dem[market, pt]

    model.con_d3 = Constraint(model.M, model.PT, rule=demand_market_con)

    # Itinerary selection constraints:
    model.AC = Set(initialize=model.I * model.M * model.Segment * model.T, filter=_filter3)

    def iti_selection( model, i, m, ap1, ap2, t, pt ):
        # if i in market_iti[m] and i in d[(ap1, ap2, t)]:
        return sum(model.x[ap1, ap2, f, t] for f in model.F) >= model.q[i, pt] / model.Dem[m, pt]

    model.con_iti_selection = Constraint(model.AC, model.PT, rule=iti_selection)
    # Restrictions on fight leg variables:
    def flight_leg_con( model, ap1, ap2, t ):
        return sum(model.x[ap1, ap2, f, t] for f in model.F) <= 1

    model.con_leg_1 = Constraint(model.Segment, model.T, rule=flight_leg_con)

    def freq_con( model, ap1, ap2 ):
        return sum(model.x[ap1, ap2, f, t] for t in model.T for f in model.F) == model.Freq[ap1, ap2]

    model.con_let_2 = Constraint(model.Segment, rule=freq_con)
    print("____" * 5)
    # for con in model.component_map(Constraint).itervalues():
    #     con.pprint()

    SOLVER_NAME = solver
    TIME_LIMIT = 60*60*2
    results = SolverFactory(SOLVER_NAME)

    if SOLVER_NAME == 'cplex':
        results.options['timelimit'] = TIME_LIMIT
    elif SOLVER_NAME == 'glpk':
        results.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':
        results.options['TimeLimit'] = TIME_LIMIT

    com = results.solve(model,tee=True)
    com.write()


    #absgap = com.solution(0).gap
    # get x results in matrix form
    df_x = pd.DataFrame(columns=list(model.Segment) ,index=model.T)
    for s in model.Segment:
        for t in model.T:
            for f in model.F:
                if model.x[s,f,t].value > 0:
                    df_x.loc[t, [s]] = f

    #df_x=df_x.reset_index()# return value  is a dataframe of new time table
    # 所有的决策变量都遍历一遍
    # for v in instance.component_objects(Var, active=True):
    #     print("Variable", v)
    #     varobject = getattr(instance, str(v))
    #     for index in varobject:
    #         print("   ", index, varobject[index].value)
    varobject = getattr(model, 'q')
    q_data = {(i,pt): varobject[(i, pt)].value for (i, pt), v in varobject.items() if  varobject[(i, pt)] !=0}
    df_q = pd.DataFrame.from_dict(q_data, orient="index", columns=["variable value"])
    varobject2 = getattr(model,'non_q')
    nonq_data = {(m,pt): varobject2[(m, pt)].value for (m, pt), v in varobject2.items() if  varobject2[(m, pt)] !=0}

    # q = list(model.q.get_values().values())
    # print('q = ', q)
    profit=model.obj()
    print('\nProfit = ', profit)
    cost=value_s(model.obj_cost())
    #revenue=value_s(model.obj_revenue())
    print('cost is:'*10,cost)
    #print('revenue is:' * 10, revenue)
    '''
    print('\nDecision Variables')
    #list_of_vars = [v.value for v in model.component_objects(ctype=Var, active=True, descend_into=True)]
    #var_names = [v.name for v in model.component_objects(ctype=Var, active=True, descend_into=True) if v.value!=0]

    # print("y=",y)
    model.obj.display()

    def pyomo_postprocess( options=None, instance=None, results=None ):
        model.x.display()

    pyomo_postprocess(None, model, results)
    for v in model.component_objects(Var, active=True):
        print("Variable component object", v)
        varobject = getattr(model, str(v))
        for index in varobject:
            if varobject[index].value != 0:
                print("   ", index, varobject[index].value)
    '''
    return df_x,q_data,profit,cost,nonq_data