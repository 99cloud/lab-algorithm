import networkx as nx
import matplotlib.pyplot as plt


# 1.创建一个图
g = nx.Graph()
g.clear     # 将图上元素清空


# 2.节点
g.add_node(1)           # 添加一个节点
g.add_node("a")
g.add_node("spam")
# g.add_nodes_from([2, 3])
nodes_list = [2, 3]     # 添加一组节点
g.add_nodes_from(nodes_list)

g.add_node("spam")          # 添加了一个名为spam的节点
g.add_nodes_from("spam")    # 添加了4个节点，名为s,p,a,m
H = nx.path_graph(10)
g.add_nodes_from(H)         # 将0~9加入了节点, 请勿使用g.add_node(H)

node_name = "spam"
g.remove_node(node_name)        # 删除节点
g.remove_nodes_from("spam")
print('g.nodes:', g.node())     # 0-9共10个节点打印出来


# 3.边
g.add_edge(1, 2)        # 添加一条边
e = (2, 3)
g.add_edge(*e)          # 直接g.add_edge(e)数据类型不对，*是将元组中的元素取出
g.add_edges_from([(0, 9), (1, 3), (1, 4)])  # 添加一组边

n = 10
H = nx.path_graph(n)
g.add_edges_from(H.edges())     # 添加了0~1,1~2 ... n-2~n-1这样的n-1条连续的边

edge_name = (0, 9)
edges_list = [(1, 3), (1, 4)]
g.remove_edge(*edge_name)       # 删除边
g.remove_edges_from(edges_list)
print('g.edges:', g.edges())


# 4.查看信息
g.number_of_nodes()     # 查看点的数量
g.number_of_edges()     # 查看边的数量
g.nodes()               # 返回所有点的信息(list)
g.edges()               # 返回所有边的信息(list中每个元素是一个tuple)
g.neighbors(1)          # 所有与1这个点相连的点的信息以列表的形式返回
print(g[1])             # 查看所有与1相连的边的属性，格式输出：{0: {}, 2: {}} 表示1和0相连的边没有设置任何属性（也就是{}没有信息），同理1和2相连的边也没有任何属性


# 5.图的属性设置
g = nx.Graph(day="Monday")
g.graph                     # {'day': 'Monday'}

g.graph['day'] = 'Tuesday'  # 修改属性
g.graph                     # {'day': 'Tuesday'}


# 6.点的属性设置
g.add_node('benz', money=10000, fuel="1.5L")
print(g.node['benz'])           # {'fuel': '1.5L', 'money': 10000}
print(g.node['benz']['money'])  # 10000
print(g.nodes(data=True))       # data默认false就是不输出属性信息，修改为true，会将节点名字和属性信息一起输出


# 7.边的属性设置
g.clear()
n = 10
H = nx.path_graph(n)
g.add_nodes_from(H)
g.add_edges_from(H.edges())
g[1][2]['color'] = 'blue'

g.add_edge(1, 2, weight=4.7)
g.add_edges_from([(3, 4), (4, 5)], color='red')
g.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
g[1][2]['weight'] = 4.7


# 8.不同类型的图（有向图Directed graphs , 重边图 Multigraphs）
# Directed graphs
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75), (1, 4, 0.3)])    # 添加带权值的边
print(DG.out_degree(1))                     # 打印结果：2 表示：找到1的出度
print(DG.out_degree(1, weight='weight'))    # 打印结果：0.8 表示：从1出去的边的权值和，这里权值是以weight属性值作为标准，如果你有一个money属性，那么也可以修改为weight='money'，那么结果就是对money求和了
print(list(DG.successors(1)))               # [2,4] 表示1的后继节点有2和4
print(list(DG.predecessors(1)))             # [3] 表示只有一个节点3有指向1的连边

# Multigraphs
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, .5), (1, 2, .75), (2, 3, .5)])
print(MG.degree(weight='weight'))   # {1: 1.25, 2: 1.75, 3: 0.5}
GG = nx.Graph()
for n, nbrs in MG.adjacency():
    for nbr, edict in nbrs.items():
        minvalue = min([d['weight'] for d in edict.values()])
        GG.add_edge(n, nbr, weight=minvalue)
print(GG.degree(weight='weight'))
print(nx.shortest_path(GG, 1, 3))     # [1, 2, 3]


# 9.图的遍历
g = nx.Graph()
g.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
for n, nbrs in g.adjacency():       # n表示每一个起始点，nbrs是一个字典，字典中的每一个元素包含了这个起始点连接的点和这两个点连边对应的属性
    print(n, nbrs)
    for nbr, eattr in nbrs.items():
        # nbr表示跟n连接的点，eattr表示这两个点连边的属性集合，这里只设置了weight，如果你还设置了color，那么就可以通过eattr['color']访问到对应的color元素
        data = eattr['weight']
        if data < 0.5:
            print('(%d, %d, %.3f)' % (n, nbr, data))


# 10.图生成和图上的一些操作
# subgraph(G, nbunch)      - induce subgraph of G on nodes in nbunch
# union(G1,G2)             - graph union
# disjoint_union(G1,G2)    - graph union assuming all nodes are different
# cartesian_product(G1,G2) - return Cartesian product graph
# compose(G1,G2)           - combine graphs identifying nodes common to both
# complement(G)            - graph complement
# create_empty_copy(G)     - return an empty copy of the same graph class
# convert_to_undirected(G) - return an undirected representation of G
# convert_to_directed(G)   - return a directed representation of G


# 11.图上分析
g = nx.Graph()
g.add_edges_from([(1, 2), (1, 3)])
g.add_node("spam")
print(list(nx.connected_components(g)))   # [[1, 2, 3], ['spam']] 表示返回g上的不同连通块
print(sorted(dict(nx.degree(g)).items(), reverse=True, key=lambda x: x[1]))

G = nx.Graph()
e = [('a', 'b', 0.3), ('b', 'c', 0.6), ('a', 'c', 0.5), ('c', 'd', 1.2)]
G.add_weighted_edges_from(e)
# 'a'可到达节点的list
print(list(nx.dfs_postorder_nodes(G, 'a')))
print(list(nx.dfs_preorder_nodes(G, 'a')))
# 获取两点间的简单路径
print(list(nx.all_simple_paths(G, 'a', 'd')))
print(list(nx.all_simple_paths(G, 'a', 'd', cutoff=2)))     # cutoff为截断常数
# 最短路径
print(nx.shortest_path(G))
print(nx.shortest_path(G, 'a', 'd'))
print(nx.has_path(G, 'a', 'd'))
# dijkstra 最短路径
print(nx.dijkstra_path(G, 'a', 'd'))
print(nx.dijkstra_path_length(G, 'a', 'd'))


# 12.图的绘制
pos = nx.spring_layout(G)
fig = plt.figure(figsize=(13, 8))
fig1 = fig.add_subplot(221)
nx.draw(G, with_labels=True, font_weight='bold', width=2)

fig2 = fig.add_subplot(222)
e = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
nx.draw_networkx_labels(G, pos, edges_label=e)
nx.draw_networkx(G, pos, with_labels=True, font_weight='bold', width=2, edge_cmap=plt.cm.Reds)
plt.axis('on')

fig3 = fig.add_subplot(223)
nx.draw_circular(DG, with_labels=True, font_weight='bold', width=2)

fig4 = fig.add_subplot(224)
nx.draw_random(DG, with_labels=True, font_weight='bold', width=2)
plt.show()
