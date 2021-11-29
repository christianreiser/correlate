import cdt
import networkx as nx

data, graph = cdt.data.load_dataset('sachs')
print(data.head())
glasso = cdt.independence.graph.Glasso()
skeleton = glasso.predict(data)
print(skeleton)
new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
print(nx.adjacency_matrix(new_skeleton)sample