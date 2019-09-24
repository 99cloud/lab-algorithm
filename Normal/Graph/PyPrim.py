import matplotlib.pyplot as plt
import networkx as nx


def prim(G, s):
    dist = {}               # dist记录到节点的最小距离
    parent = {}             # parent记录最小生成树的双亲表
    q = list(G.nodes())     # q包含所有未被生成树覆盖的节点
    max_dist = 9999.99      # max_dist表示正无穷，即两节点不邻接

    # 初始化数据
    # 所有节点的最小距离设为max_dist，父节点设为None
    for v in G.nodes():
        dist[v] = max_dist
        parent[v] = None
    # 到开始节点s的距离设为0
    dist[s] = 0

    # 不断从q中取出“最近”的节点加入最小生成树
    # 当q为空时停止循环，算法结束
    while q:
        # 取出“最近”的节点u，把u加入最小生成树
        u = q[0]
        for v in q:
            if dist[v] < dist[u]:
                u = v
        q.remove(u)

        # 更新u的邻接节点的最小距离
        for v in G.adj[u]:
            if (v in q) and (G[u][v]['weight'] < dist[v]):
                parent[v] = u
                dist[v] = G[u][v]['weight']
    # 算法结束，以双亲表的形式返回最小生成树
    return parent


def draw(G, edge_style='solid', edge_colors='k', edge_width=2, tree_colors='y'):
    pos = nx.spring_layout(G)
    nx.draw(G, pos,
            arrows=True,
            with_labels=True,
            node_size=1000,
            font_size=23,
            font_family='times new roman',
            font_width='bold',
            nodelist=G.nodes(),
            style=edge_style,
            edge_color=edge_colors,
            width=edge_width,
            node_color=tree_colors,
            alpha=0.5)
    plt.show()


if __name__ == '__main__':
    G_data = [(1, 2, 1.3), (1, 3, 2.1), (1, 4, 0.9), (1, 5, 0.7), (1, 6, 1.8), (1, 7, 2.0), (1, 8, 1.8), (2, 3, 0.9),
              (2, 4, 1.8), (2, 5, 1.2), (2, 6, 2.8), (2, 7, 2.3), (2, 8, 1.1), (3, 4, 2.6), (3, 5, 1.7), (3, 6, 2.5),
              (3, 7, 1.9), (3, 8, 1.0), (4, 5, 0.7), (4, 6, 1.6), (4, 7, 1.5), (4, 8, 0.9), (5, 6, 0.9), (5, 7, 1.1),
              (5, 8, 0.8), (6, 7, 0.6), (6, 8, 1.0), (7, 8, 0.5)]

    G = nx.Graph()
    G.add_weighted_edges_from(G_data)   # 添加赋权边
    tree = prim(G, 1)                   # 获取最小生成树
    print('Minimum Spanning Tree: ', tree)

    tree_edges = [(u, v) for u, v in tree.items()]  # 将生成树转成边的格式
    G.add_edges_from(tree_edges)
    G.remove_node(None)

    TG = nx.Graph()
    TG.add_edges_from(tree_edges)
    TG.remove_node(None)

    tree_degree = []
    tree_colors = []
    for i in G.node:
        tree_degree.append(TG.degree[i])
        if TG.degree[i] >= 3:
            tree_colors.append('r')
        elif TG.degree[i] >= 2:
            tree_colors.append('g')
        else:
            tree_colors.append('y')
    print('Tree Degree:', tree_degree)

    tree_edges_zf = tree_edges + [(v, u) for u, v in tree.items()]

    edge_colors = ['red' if edge in tree_edges_zf else 'black' for edge in G.edges]
    edge_style = ['solid' if edge in tree_edges_zf else 'dashed' for edge in G.edges]
    edge_width = [3 if edge in tree_edges_zf else 1.5 for edge in G.edges]

    draw(G, edge_style, edge_colors, edge_width, tree_colors)
