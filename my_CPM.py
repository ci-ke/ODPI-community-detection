import networkx as nx
from collections import defaultdict


def get_percolated_cliques(G, k):
    percolation_graph = nx.Graph()
    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= k]
    percolation_graph.add_nodes_from(cliques)

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    def get_adjacent_cliques(clique, membership_dict):
        adjacent_cliques = set()
        for n in clique:
            for adj_clique in membership_dict[n]:
                if clique != adj_clique:
                    adjacent_cliques.add(adj_clique)
        return adjacent_cliques

    # For each clique, see which adjacent cliques percolate
    for clique in cliques:
        for adj_clique in get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                percolation_graph.add_edge(clique, adj_clique)

    # print 'percolation graph nodes:', percolation_graph.nodes()
    # print 'percolation graph edges:', percolation_graph.edges()

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(percolation_graph):
        yield (frozenset.union(*component))


def get_community_nodes_dict(G, k=3):
    comm_list = list(get_percolated_cliques(G, k))
    community_nodes_dict = {}
    community = 1
    for comm_nodes in comm_list:
        community_nodes_dict[community] = list(comm_nodes)
        community += 1
    return community_nodes_dict


if __name__ == '__main__':
    G = nx.read_gml("./datasets/known/karate.gml", label="id")
    community_nodes_dict = get_community_nodes_dict(G)
    for key, value in list(community_nodes_dict.items()):
        print(value)
