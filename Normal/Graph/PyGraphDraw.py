import networkx as nx
import matplotlib.pyplot as plt
import pylab

G = nx.DiGraph()

G.add_edges_from([('A', 'B'), ('C', 'D'), ('G', 'D')], weight=1)
G.add_edges_from([('D', 'A'), ('D', 'E'), ('B', 'D'), ('D', 'E')], weight=2)
G.add_edges_from([('B', 'C'), ('E', 'F')], weight=3)
G.add_edges_from([('C', 'F')], weight=4)

val_map = {'A': 1.0,
           'D': 0.8,
           'H': 0.0}

values = [val_map.get(node, 0.1) for node in G.nodes()]
print(values)
edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
print(edge_labels)
red_edges = [('C', 'D'), ('D', 'A')]
edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
print(edge_colors)

pos = nx.spring_layout(G)
nx.draw_networkx_labels(G, pos, font_size=15, font_color='w', font_family='Arial')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=15, font_family='Arial')
nx.draw(G, pos, node_color=values, node_size=1000, edge_color=edge_colors, width=2, edge_cmap=plt.cm.Reds)
pylab.show()