import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph


n_nodes = 6



neurons = np.random.random(size=(n_nodes,))
weights = np.random.random(size=(n_nodes, n_nodes))
# weights = np.asarray([[.1,.2,.5], [.7, .8,.5], [.1, 0, 0]])

lengths = np.random.randint(low=2, high=8, size=(n_nodes, n_nodes))
print(neurons)
print(weights)
print(lengths)
dot = Digraph(engine='neato')

for idx, node in enumerate(neurons):
    minvis = 0.05
    pos = str(hex(max(int(255*node), int(255*minvis))))[2:]
    print(pos)
    dot.node(name=str(idx), 
             label=f"{node:.2f}", 
             # fixedsize=False, 
             # arrowsize=2, 
             color=f"#000000{pos}")
# edges
for idx, start in enumerate(weights):
    for jdx, end in enumerate(start):
        if end:# and idx != jdx:
            minvis = 0.05
            pos = str(hex(max(int(255*end), int(255*minvis))))[2:]
            dot.edge(tail_name=str(idx), 
                     head_name=str(jdx), 
                     color=f"#000000{pos}",
                     len=str(lengths[idx][jdx]))

dot.render("test.png", format="png", view=True)
plt.plot(dot)