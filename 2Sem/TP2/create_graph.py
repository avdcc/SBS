#city_name,description,cause_of_incident,from_road,to_road,affected_roads,incident_category_desc,magnitude_of_delay_desc,length_in_meters,delay_in_seconds,incident_date
import sys
import functools as f
#import netgraph
#import matplotlib.pyplot as plt; plt.ion()
import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
import re
import unidecode
from datetime import datetime, timedelta
from numpy import round

#####################
#     DEFINES       #
#####################
OUTPUT = 'Braga/Traffic_Flow_Braga_Until_20190228.csv'
#OUTPUT = 'Guimaraes/tfw.csv'
NUM_PARAMETROS_SEM_RUAS = 20
DATA_ind = 9 
#####################


def readFile(name_file):
    with open(name_file,'r') as fd:
        #return list(map( lambda x: x.split(','),fd.read().strip("'").strip("\"").split('\n')))
        return list(map( lambda x: x.split(','),fd.read().lower().replace('"','').split('\n')))

def writeFile(name_file,table):
    with open(name_file,'w') as fd:
        fd.write( '\n'.join(list(map(lambda x: ';'.join(x),table))) )

def printTable(table):
        print( '\n'.join(list(map(lambda x: ';'.join(x),table))) )

def testa_dataSet():
    lista = readFile(cidade + 'ti.csv')[:-1]
    # 11, 12, 13, 14
    dic = {}
    dic[11] = []
    dic[12] = []
    dic[13] = []
    dic[14] = []
    #return set(map(lambda x: len(x),lista))
    for i,l in enumerate(lista): 
        dic[len(l)].append(i)
    return dic

if (len(sys.argv) > 1):
    cidade = sys.argv[1] + '/'
else:
    #cidade = 'Guimaraes/'
    cidade = 'Braga/'
    #cidade = 'Porto/'

#print(testa_dataSet())
#lista = readFile(cidade + 'ti.csv')[1:-1]
lista = readFile(cidade + 'Traffic_Incidents_Braga_Until_20190228.csv')[1:-1]
#lista = readFile(cidade + 'Traffic_Incidents_Porto_Until_20190228.csv')[1:-1]
#print(lista[3][3])

def trata_rua(rua):
    # retira os ()
    # tira o espaco antes
    #return unidecode.unidecode(re.sub(' \(.*?\)', '', rua).replace(" ",""))
    return unidecode.unidecode(re.sub('\(.*', '',re.sub('\(.*\)', '', re.sub('\[.*?\]','',rua.replace(" ","")) )) )
    


#    3      |    4    |        5       |            7            |         8        |        9         |      10      
# from_road | to_road | affected_roads | magnitude_of_delay_desc | length_in_meters | delay_in_seconds | incident_date
# string    | string  |  | magnitude_of_delay_desc | length_in_meters | delay_in_seconds | incident_date

from_road = list(map(lambda x: trata_rua(x[3]),lista))
#print(len(from_road))

#print(from_road)

to_road = list(map(lambda x: trata_rua(x[4]),lista))
#print(len(to_road))
#print(to_road)
#print(set(to_road))

affected_roads = list(map(lambda x: list(map(lambda y: trata_rua(y),x[5:-5])) if x[5:-5]!= [''] else [],lista))
#print(len(affected_roads))
#print(affected_roads)

def rename_mag(string):
    if(string == 'minor'): return 1
    if(string == 'moderate'): return 2
    if(string == 'major'): return 3
    else: return 0

# -6 5
# -5 6
# -4 7
magnitude_of_delay_desc = list(map(lambda x: rename_mag(x[-4]),lista))
#magnitude_of_delay_desc = list(map(lambda x: x[-4],lista))
#print(len(magnitude_of_delay_desc))
#print(magnitude_of_delay_desc)
#print(set(magnitude_of_delay_desc))

length_in_meters = list(map(lambda x: int(x[-3]),lista))
#print(len(length_in_meters))
#print(length_in_meters)
#print(sorted(list(set(length_in_meters))))

delay_in_seconds = list(map(lambda x: int(x[-2]),lista))
#print(len(delay_in_seconds))
#print(delay_in_seconds)

incident_date = list(map(lambda x: x[-1],lista))
#print(len(incident_date))
#print()
#print(len(lista))
#print('\n'.join(incident_date))

def separa_rua(rua,i):
    res = ''
    if(len(rua[i].split('/')) > 1):
        res = rua[i].split('/')[1]
        rua[i] = rua[i].split('/')[0]
    return res


for i in range(len(lista)):
    from_road_aux = separa_rua(from_road,i)
    to_road_aux = separa_rua(to_road,i)
    if(from_road != '' or to_road != ''):
        from_road.append(from_road_aux) if from_road_aux != '' else from_road.append(from_road[i]) 
        to_road.append(to_road_aux) if to_road_aux != '' else to_road.append(to_road[i]) 

        affected_roads.append(affected_roads[i])
        magnitude_of_delay_desc.append(magnitude_of_delay_desc[i])
        length_in_meters.append(length_in_meters[i])
        delay_in_seconds.append(delay_in_seconds[i])
        incident_date.append(incident_date[i]) 

#print(len(from_road))
#print(len(to_road))
#print(len(affected_roads))
#print(len(magnitude_of_delay_desc))
#print(len(length_in_meters))
#print(len(delay_in_seconds))
#print(len(incident_date))


def lltl(ll):
    return f.reduce(lambda x,y: x + y , ll)
# todas as ruas
#def ruas(): return list(set( from_road + to_road + affected_roads ))
def ruas(): return list(filter(lambda x: x!='',list(set( from_road + to_road + lltl(affected_roads) ))))

def create_graph():
    graph = {}
    for r in ruas():
        graph[r] = set()
    return graph
#print(create_graph())

def generate_graph():
    graph = create_graph()
    for x in range(1,len(lista)):
        if (to_road[x] != '') : 
            graph[ from_road[x] ].add(to_road[x])
            graph[ to_road[x] ].add(from_road[x])
        graph[ from_road[x] ].update(affected_roads[x])
    return graph

#print(generate_graph()[from_road[7]])
#print(generate_graph())
#graph = generate_graph()
#keys = list(graph.keys())
#print( str(keys[1])+ ' : ' +  str(graph[keys[1]]) )

def date(): return list(set(incident_date))

def acidentes_date_func(date,f):
    res = []
    for i,l in enumerate(incident_date):
        if( f(l) ): res.append(i)
    return res

def acidentes_date(date):
    return acidentes_date_func(date,lambda x: x == date)

def try_int(x):
    try:
        return int(x)
    except:
        return int(round(float(x)))

def date_to_tuple(date):
    try:
        year , month, day = date[0].split('-')
        hour, minute, second = date[1].split(':')
    except:
        date = date.split(' ')
        year , month, day = date[0].split('-')
        hour, minute, second = date[1].split(':')
    #year , month, day = date[0].split('-')
    #hour, minute, second = date[1].split(':')
    return tuple(map(try_int,[year,month,day,hour,minute,second]))


def date_to_datetime(date):
    try:
        return datetime.strptime(date,'%Y-%m-%d %H:%M:%S.%f')
    except:
        return datetime.strptime(date,'%Y-%m-%d %H:%M:%S')

#'2018-07-24 14:58:54.118000'
def menor_date(date1, date2):
    return date_to_tuple(date1) <= date_to_tuple(date2)

def acidentes_menor_date(date):
    return acidentes_date_func(date,lambda x: menor_date(date.split(' '),x.split(' ')))

def fator_prop(dist):
   return dist

def ordena(l):
    return sorted([[ date_to_datetime(incident_date[i]),
           from_road[i],
           to_road[i],
           affected_roads[i],
           delay_in_seconds[i]] for i in l],key= lambda x: x[0],reverse=True)
           #magnitude_of_delay_desc[i]] for i in l],key= lambda x: x[0],reverse=True)


def new_date(date,range_date =12):
    #hour, minute, second = list(map(try_int,date[1].split(':')))
    #year, month, day = list(map(int,date[0].split('-')))
    #return (datetime(year,month,day,hour,minute,second) - timedelta(hours = range_date) ).isoformat(' ').split(' ')
    return date_to_datetime(date) - timedelta(hours = range_date) 
            

def filter_date(l,date,range_date):
    res = []
    for x in l:
        if (x[0] > new_date(date,range_date)):
            res.append(x)
    return res
    #return filter(lambda x: x[0] > new_date(date,range_date),l)

#(magnitude_of_delay_desc[i], length_in_meters[i], delay_in_seconds[i]))
def calc_peso_foco(peso):
    return sum(list(peso)) 

def update_peso(peso):
    return (peso[0]-1,peso[1]-100,peso[2])
    

def forEach(f,l):
    for x in l: f(x)

def find_replace(foco,data,peso,table):
    # !!!!!!!!!!!!!!
    # MUDAR OS INDICES
    # !!!!!!!!!!!!!!
    for i,x in enumerate(table):
        if((x[0] == foco) and (x[1] == data)):
            table[i][-1] += peso
    return table


def write_file(date,foco,peso):
    # !!!!!!!!!!!!!!
    # MUDAR O PATH
    # !!!!!!!!!!!!!!
    table = readFile( cidade +'/< ... >')
    table = find_replace(foco,data,peso,table)
    writeFile( cidade +'/< ... >',table)

def write_file_func(date):
    return lambda x: write_file(date,x)

def graph_date_foco(date,graph,foco,acidente):
    write_file(date,foco,calc_peso_foco(acidente))

    acidente = update_peso(acidente)
    if((graph[ foco ] != set()) and (acidente[1] > 10) and (acidente[0] > 0)):
        forEach(lambda x: graph_date_foco(date,
                                          graph,
                                          x,
                                          acidente),graph[ foco ])


def graph_date(date,graph):
    for i in acidentes_date(date):
        graph_date_foco(date,
                        graph,
                        from_road[i],
                        (magnitude_of_delay_desc[i],
                        length_in_meters[i],
                        delay_in_seconds[i]))

    

def show_graph():
    g = generate_graph()
    ng = nx.DiGraph(g)
    nx.draw(ng)
    plt.show()

def print_conect():
    g = generate_graph()
    res = set()
    for x in g.keys():
        #print(str(x) + ': ' + str(list(len(g[x]))))
        print(x + '---> ' + str( len(list(g[x])) ))
        #res.add()

def calc_weight(date1,date2,d):
    #date1 = date_to_datetime(date1)
    # diff hours
    #print(date2)
    diff = date1 - date2 if (date1 > date2) else date2 - date1
    diff = diff.total_seconds()
    diff = round(float(diff)/(60*60))
    #if(d == 3): print(diff)
    return int(max( round(d/(diff + 1)) ,0))

def street_weight(l,ruas,date):
    calculadas = []
    res = []
    for row in l:
        if(len(calculadas)==len(ruas)):
            break
        else:
            weight = calc_weight(date,row[0],row[-1])
            #print(row[-1])
            #weight = row[-1]
            if(row[1] not in calculadas):
                calculadas.append(row[1])
                res.append((row[1],weight))
            if(row[2] not in calculadas):
                calculadas.append(row[2])
                res.append((row[1],weight))
            for street in row[3]:
                if(street not in calculadas):
                    calculadas.append(street)
                    res.append((street,weight))

    return res
            


#print('numero de ruas: ' + str(len(list(ruas()))))
#show_graph()
#print_conect()
#print(list(map(lambda x: ' '.join(x),ordena(acidentes_menor_date('2018-07-24 14:58:54.118000')))))
#forEach(print,ordena(acidentes_menor_date('2018-07-24 14:58:54.118000')))
#forEach(print,filter_date(ordena(range(len(incident_date))),'2019-02-28 19:45:01.855000',12))
#print(new_date('2018-07-24 14:58:54.118000'.split(' '),99907))
#print(max(acidentes_menor_date('2018-07-24 14:58:54.118000')))
#print(len(lista))

# date from_road to_road affected_roads magnitude_of_delay_desc
#DataSet = filter_date(ordena(range(len(incident_date))),'2019-02-28 19:45:01.855000',12)
#['road_name', 'functional_road_class_desc', 'current_speed', 'free_flow_speed', 'speed_diff', 'current_travel_time', 'free_flow_travel_time', 'time_diff', 'creation_date', 'datecomplete', 'creation_time']
Output = readFile(OUTPUT) 
DATAS_OUTPUT = set()
#DATAS_OUTPUT.add('2019-02-18 09:30:00')
for x in Output[1:-1]:
    #DATAS_OUTPUT.add(x[9])
    DATAS_OUTPUT.add(x[10])

# print(Output[0:3])
l_ruas = ruas()
#print('[' + 'Data' +','+ ','.join(l_ruas) + ']')
print('Data' +','+ ','.join(l_ruas))
DataSet = ordena(range(len(incident_date)))
#print(len(list(DATAS_OUTPUT)))
#for x in list(DATAS_OUTPUT):
#    print(x)
    #try:
    #    date_to_datetime(x)
    #except:
    #    print(x)
for data in list(DATAS_OUTPUT):
    date = date_to_datetime(data)
    sw = street_weight(DataSet, l_ruas, date)
    print(data + ',' + ','.join(list(map(lambda x: str(x[1]),sw))))
# 21
#print(len(Output[0]))
#printTable(Output)
# 244
# l_ruas = ruas()
# #print(len(l_ruas))
# ruas_ind = dict(list(map(lambda x : (x[1],x[0]),list(enumerate(l_ruas)))))
# #print(ruas_ind)
# #print(Output)
# nl = len(Output[1])
# #if(nl == NUM_PARAMETROS_SEM_RUAS):
# if(True):
#    # criar uma 
#    #print(Output)
#    Output[0] += l_ruas
#    Output[1:] = list(map(lambda x: x + len(l_ruas)*['0'],Output[1:]))

   #printTable(Output)
#print(len(Output[1]))
#print(Output[1])
#print(DataSet[1])
# data_sw = {}
# #DataSet = ordena(range(len(incident_date)))
# #print(DataSet)
# #table_res = Output
# for i,row in enumerate(Output):
#    if(i == 0) or (i == len(Output)-1): continue
#    if(len(row) < DATA_ind): continue
#    #print(DataSet)
#    #print(row[Data_ind])
#    data = row[10]
#    #print(data)
#    date = date_to_datetime(data)

#    if(data not in data_sw.keys()):
#        #print(DataSet)
#        sw = street_weight(DataSet,l_ruas,date)
#        data_sw[data] = sw
#    else:
#        sw = data_sw[data]
#    #print(date)
#    # [(street,weight) , ...]

#    #print(sw)
#    for x in sw:
#        if(x[0] != ''):
#            #None
#            # 265
#            #print(len(Output[i]))
#            #print(ruas_ind[x[0]]+NUM_PARAMETROS_SEM_RUAS+1)
#            try:
#                Output[i][ruas_ind[x[0]]+NUM_PARAMETROS_SEM_RUAS+1] = str(x[1])
#            except:
#                None

           #table_res[ruas_ind[x[0]]+NUM_PARAMETROS_SEM_RUAS+1] = str(x[1])
           #print(x[0])
           #print('----')

#forEach(lambda x: print(';'.join(x)),table_res)
#writeFile('testeout.csv',Output)
#writeFile(OUTPUT,Output)
#printTable(Output)
#
#
##for x in DataSet:
#
#
#
#
#
##print()
##print()
##print()
#
#

