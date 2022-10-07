# -*- coding: utf-8 -*-

'''
基于LPA稳定的标签传播
1）异步更新
2）更新顺序因节点的重要性以此排序
3）当多个标签相同时，将节点的影响值引入，从而避免因随机选择而得不到一个稳定的收敛状态
'''
import numpy as np
import networkx as nx
from collections import defaultdict


def find_communities(G, T, r):
    """
    Speaker-Listener Label Propagation Algorithm (SLPA)
    see http://arxiv.org/abs/1109.5720
    """

    ##Stage 1: Initialization
    memory = {i: {i: 1} for i in G.nodes()}

    ##Stage 2: Evolution
    for t in range(T):

        listenersOrder = list(G.nodes())
        np.random.shuffle(listenersOrder)

        for listener in listenersOrder:
            speakers = list(G[listener].keys())
            if len(speakers) == 0:
                continue

            labels = defaultdict(int)

            for j, speaker in enumerate(speakers):
                # Speaker Rule
                total = float(sum(memory[speaker].values()))
                labels[
                    list(memory[speaker].keys())[
                        np.random.multinomial(
                            1, [freq / total for freq in list(memory[speaker].values())]
                        ).argmax()
                    ]
                ] += 1

            # Listener Rule
            acceptedLabel = max(labels, key=labels.get)

            # Update listener memory
            if acceptedLabel in memory[listener]:
                memory[listener][acceptedLabel] += 1
            else:
                memory[listener][acceptedLabel] = 1

    ## Stage 3:
    for node, mem in memory.items():
        for label, freq in list(mem.items()):
            if freq / float(T + 1) < r:
                del mem[label]

    # Find nodes membership
    communities = {}
    for node, mem in memory.items():
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    # Remove nested communities
    nestedCommunities = set()
    keys = list(communities.keys())
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i + 1 :]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nestedCommunities.add(label0)
            elif comm0.issuperset(comm1):
                nestedCommunities.add(label1)

    for comm in nestedCommunities:
        del communities[comm]

    return communities


def get_community_nodes_dict(G, num_iterations=20, threshold=0.05):
    comms = find_communities(G, num_iterations, threshold)
    community_nodes_dict = {}
    community = 1
    for ket, comm_nodes in list(comms.items()):
        community_nodes_dict[community] = list(comm_nodes)
        community += 1
    return community_nodes_dict


if __name__ == "__main__":
    G = nx.read_gml("./datasets/karate.gml", label="id")
    # 默认边的权重为1.0
    for edge in G.edges:
        if G[edge[0]][edge[1]].get('weight', -1000) == -1000:
            G[edge[0]][edge[1]]['weight'] = 1.0
    community_nodes_dict = get_community_nodes_dict(G, threshold=0.1)
    for key, value in list(community_nodes_dict.items()):
        print(value)
