# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from biopandas.pdb import PandasPdb
import random

import networkx as nx

import Constants
from Dataset.Dataset import load_dataset
import matplotlib.pyplot as plt

kwargs = {
    'prot_ids': ['P83847', ],
    'session': 'selected'
}

dataset = load_dataset(root=Constants.ROOT, **kwargs)
protein = dataset[0]
print(protein)
node_coords = protein.pos
edges = protein.edge_index


exit()

# x, y, z = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]
# # print(x.shape)
# # print(y.shape)
# # print(z.shape)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# ax.scatter3D(x, y, z, 'gray')
# # ax.plot3D(x, y, z, 'gray')
#
# fig.tight_layout()
# plt.show()




# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from biopandas.pdb import PandasPdb
import networkx as nx
import numpy as np

import Constants
from Dataset.Dataset import load_dataset
import matplotlib.pyplot as plt

kwargs = {
    'prot_ids': ['A0A023FBW4', ],
    'session': 'selected'
}

dataset = load_dataset(root=Constants.ROOT, **kwargs)
protein = dataset[0]

print(protein)



exit()


def plot_residues(protein, add_edges=False, limit=25):
    node_coords = protein.pos.numpy()

    limit = len(node_coords)

    if add_edges:
        indicies = random.sample(range(0, limit), 25)
    else:
        indicies = random.sample(range(0, limit), limit)

    edges = protein.edge_index.numpy()
    some_edges = []
    edges = [i for i in zip(edges[0], edges[1])]
    for i, j in edges:
        if i in indicies and j in indicies:
            some_edges.append(([node_coords[i][0], node_coords[j][0]],
                               [node_coords[i][1], node_coords[j][1]],
                               [node_coords[i][2], node_coords[j][2]]))

    node_coords = np.array([node_coords[i] for i in indicies])

    x, y, z = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]
    # # print(x.shape)
    # # print(y.shape)
    # # print(z.shape)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z)

    if add_edges:
        for x in some_edges:
            ax.plot3D(x[0], x[1], x[2])# , 'gray')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # fig.tight_layout()
    plt.title("Some protein")

    plt.show()


plot_residues(protein)