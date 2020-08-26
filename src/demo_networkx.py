import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

neurons = np.asarray([.3, .5, .6])

weights = np.asarray([[0,.2,.5], [.7,0,.5], [.1, 0, 0]])
n_nodes = neurons.size
lengths = [[0,4,3], [2,0,2], [7,1,0]]


G=nx.generators.directed.gn_graph(n_nodes)
pos=nx.kamada_kawai_layout(G, dist=lengths) # positions for all nodes

# nodes

nx.draw_networkx_nodes(G,
                       pos,
                       nodelist=range(n_nodes),
                       node_color='r',
                       node_size=500,
                       alpha=0.8)


# edges
for idx, start in enumerate(weights):
    for jdx, end in enumerate(start):
        if end:# and idx != jdx:
            nx.draw_networkx_edges(G, 
                                   pos,
                                   arrows=True, 
                                   edgelist=[(idx, jdx)],
                                   width=8,
                                   alpha=end,
                                   edgecolor='r',
                                   connectionstyle='arc3,rad=0.2')


# nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
# nx.draw_networkx_edges(G,pos,
#                        edgelist=[(0,1),(1,2),(2,3),(3,0)],
#                        width=8,alpha=0.5,edge_color='r')
# nx.draw_networkx_edges(G,pos,
#                        edgelist=[(4,5),(5,6),(6,7),(7,4)],
#                        width=8,alpha=0.5,edge_color='b')


# some math labels
labels={}
labels[0]=r'$a$'
labels[1]=r'$b$'
labels[2]=r'$c$'
nx.draw_networkx_labels(G,pos,labels,font_size=16)

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display


# gph = nx.Graph()

# neurons = np.asarray([.3, .5, .6])

# weights = np.asarray([[.1,.2,.5], [.7,.4,.5], [.1, 0, .2]])
  
# for idx, start in enumerate(weights):
#   for jdx, end in enumerate(start):
#       print(idx, jdx)
#       gph.add_edge(idx, jdx)


# # gph.add_edge(1, 2) 
# # gph.add_edge(2, 3) 
# # gph.add_edge(3, 4) 
# # gph.add_edge(1, 4) 
# # gph.add_edge(1, 5)
  
# nx.draw(gph, node_size=500, node_color=neurons)  # draw_[how]()

# plt.show()