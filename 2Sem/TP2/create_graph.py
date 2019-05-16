#city_name,description,cause_of_incident,from_road,to_road,affected_roads,incident_category_desc,magnitude_of_delay_desc,length_in_meters,delay_in_seconds,incident_date
import sys
import functools as f

def readFile(name_file):
    with open(name_file,'r') as fd:
        return list(map( lambda x: x.split(','),fd.read().strip("'").strip('"').split('\n')))

def writeFile(name_file,table):
    with open(name_file,'w') as fd:
        fd.write( '\n'.join(list(map(lambda x: ';'.join(x),table))) )


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
    cidade = 'Guimaraes/'

#print(testa_dataSet())
lista = readFile(cidade + 'ti.csv')[1:-1]
#print(lista[3][3])


#    3      |    4    |        5       |            7            |         8        |        9         |      10      
# from_road | to_road | affected_roads | magnitude_of_delay_desc | length_in_meters | delay_in_seconds | incident_date
# string    | string  |  | magnitude_of_delay_desc | length_in_meters | delay_in_seconds | incident_date

from_road = list(map(lambda x: x[3],lista))
#print(from_road)

to_road = list(map(lambda x: x[4],lista))
#print(to_road)
#print(set(to_road))

affected_roads = list(map(lambda x: x[5:-5] if x[5:-5]!= [''] else [],lista))
#print(affected_roads)

def rename_mag(string):
    if(string == 'Minor'): return 1
    if(string == 'Moderate'): return 2
    if(string == 'Major'): return 3
    else: return 0

# -6 5
# -5 6
# -4 7
#magnitude_of_delay_desc = list(map(lambda x: rename_mag(x[-4]),lista))
magnitude_of_delay_desc = list(map(lambda x: x[-4],lista))
#print(magnitude_of_delay_desc)
#print(set(magnitude_of_delay_desc))

length_in_meters = list(map(lambda x: int(x[-3]),lista))
#print(length_in_meters)
#print(sorted(list(set(length_in_meters))))

delay_in_seconds = list(map(lambda x: int(x[-2]),lista))
#print(delay_in_seconds)

incident_date = list(map(lambda x: x[-1],lista))
#print(incident_date)

def lltl(ll):
    return f.reduce(lambda x,y: x + y , ll)
# todas as ruas
#def ruas(): return list(set( from_road + to_road + affected_roads ))
def ruas(): return filter(lambda x: x!='',list(set( from_road + to_road + lltl(affected_roads) )))
             
def create_graph():
    graph = {}
    for r in ruas():
        graph[r] = set()
    return graph
#print(create_graph())

def generate_graph():
    graph = create_graph()
    for x in range(1,len(lista)):
        if (to_road[x] != '') : graph[ from_road[x] ].add(to_road[x])
        graph[ from_road[x] ].update(affected_roads[x])
    return graph

#print(generate_graph()[from_road[7]])
#print(generate_graph())
#graph = generate_graph()
#keys = list(graph.keys())
#print( str(keys[1])+ ' : ' +  str(graph[keys[1]]) )

def date(): return list(set(incident_date))

def acidentes_date(date):
    #return list(filter(lambda x: x[10] == hora,lista))
    res = []
    for i,l in enumerate(lista):
        if( x[10] == date ): res.append(i)


def fator_prop(dist):
   return dist


#(magnitude_of_delay_desc[i], length_in_meters[i], delay_in_seconds[i]))
def calc_peso_foco(peso):
    return sum(list(peso)) 

def update_peso(peso):
    return (peso[0]-1,peso[1]-100,peso[2])
    

def forEach(f,l):
    for x in f: f(x)

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

    



print()
print()
print()
