"""
Module designed to extract a network (as networkx graph) from a skeletonized array (from scikit-image skeletonize function for instance).
"""
import numpy as np
from shapely.geometry import LineString
import networkx as nx

def network_from_skeleton(array: np.ndarray, axesswap = True):
    """"
    Return a weighted networkx graph from a binary array (0s and 1s), where 1s consist of a serie of connected 1pixel-wide lines (= medial axis/skeleton).
    A good example input is the output array of skeletonize() function from scikit-image package (after transformation of True/False array to 1s and 0s array).
    
    Parameters
    ----------
    array : ndarray, 2D
        skeleton array from wich to extract the network.
    axesswap : bool, by default True
        Swap axes.
        Function is initially designed to extract a network from an array stemming from an image.
        For the network coordinates (see pos attribute) to match the image, a swapping of axes is required.
    
    Returns
    -------
        graph : networkx.classes.multigraph.MultiGraph
        
    Examples
    --------
    >>> example = np.array([
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0],
        [0,1,1,1,0,0,1,0,1,0,1,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0],
        [0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0],
        [1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
        [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,0,1,0,0,1,1,1,0,0,0,0,0,0],
        [0,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1],
        [0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,0,1],
        [0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1],
        [0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0],
        [0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1],
    ])
    >>> G = network_from_skeleton(example)
    >>> pos = nx.get_node_attributes(G, 'pos')
        fig, ax = plt.subplots()
        plt.imshow(example, cmap=plt.cm.binary)
        nx.draw_networkx_nodes(G, pos, node_size = 50, ax = ax)
        nx.draw_networkx_edges(G, pos, edge_color = "red", ax = ax)

    """
    points, nodes = points_and_nodes_from_skeleton(array)
    if axesswap:
        points, nodes = [axes_swap(i) for i in [points, nodes]]
    edges = edges_from_points_and_nodes(points, nodes)
    graph = graph_from_edges(edges)
    return graph

def points_and_nodes_from_skeleton(array: np.ndarray):
    """"
    Return an ordered list of points and a list of nodes from a binary array (0s and 1s), where 1s consist of a serie of connected 1pixel-wide lines (= medial axis/skeleton).
    A good example input is the output array of skeletonize() function from scikit-image package (after transformation of True/False array to 1s and 0s array).
    """
    if array.ndim != 2:
        raise ValueError("array/skeleton must be 2D")
    if set(list(array.reshape(array.size))) != {0, 1}:
        raise ValueError("array must be binary (0s and 1s only)")
        
    # add a border of zeros so that every 1-pixel has neighboring pixels
    array = np.pad(array, pad_width=1, mode='constant')
    
    points = []
    nodes = []
    nodes_to_do = []
    to_avoid = []
    nodes_and_its_connections = {}
    
    # count the number of ones. Process is finished once count = 0
    array_as_list = list(array.reshape(array.size))
    
    while array_as_list.count(1) != 0:
        # select an extremity (dead-end) as starting point
        for i in range(1, array.shape[0]-1):
            for j in range(1, array.shape[1]-1):
                if (array[i, j] == 1) and (sum(window(array, i, j)) <= 2):
                    row = i
                    col = j
                    break
            if (array[i, j] == 1) and (sum(window(array, i, j)) <= 2):
                break
        
        # get neighboring informations
        cross_indexes, cross_values, diag_indexes, diag_values = neighbors_index_and_value(array, row, col)
        
        # the number of paths defines whether or not a node needs to be created
        paths_number = sum([value == 1 for value in cross_values] + [value == 1 for value in diag_values])
        
        point = (row, col)
        points.append(point)
        array[point] = 999 # point done 
        nodes.append(point)
        nodes_and_its_connections[point] = [cross_indexes[i] for i in range(len(cross_indexes)) if cross_values[i] == 1] + [diag_indexes[i] for i in range(len(diag_indexes)) if diag_values[i] == 1]

        while (paths_number != 0) or nodes_to_do:
            if (paths_number == 0): # reached a dead-end, go back to a node not done yet
                row, col = nodes_to_do[-1]
                nodes_to_do = nodes_to_do[:-1]
            else:
                if paths_number > 1: # create a node
                    nodes.append((row, col))
                    nodes_to_do.append((row, col))
                    nodes_and_its_connections[(row, col)] = [cross_indexes[i] for i in range(len(cross_indexes)) if cross_values[i] ==1] + [diag_indexes[i] for i in range(len(diag_indexes)) if diag_values[i] ==1]
                    #remember neighbors that are part of another path than the one folowed here after
                    to_avoid = [tuple(i) for i in np.array(cross_indexes)[cross_values == 1]]
                
                # go to next point
                if any([value == 1 for value in cross_values]):
                    relative_next_cell_position = np.argwhere(np.array(cross_values) == 1)[0][0]
                    row, col = cross_indexes[relative_next_cell_position] 
                else:
                    relative_next_cell_position = np.argwhere(np.array(diag_values) == 1)[0][0]
                    row, col = diag_indexes[relative_next_cell_position]
            
            array[point] = 999 # point done       
            point = (row, col) # assign next point
            points.append(point)
            
            # get neighboring informations
            cross_indexes, cross_values, diag_indexes, diag_values = neighbors_index_and_value(array, row, col)
         
            # preventing corners to be confused with nodes
            potential_pos_in_cross = np.array(cross_indexes)[cross_values != 0]
            potential_pos_in_cross = [tuple(i) for i in potential_pos_in_cross]
            potential_pos_in_diag = np.array(diag_indexes)[diag_values != 0]
            potential_pos_in_diag = [tuple(i) for i in potential_pos_in_diag]
            for i in potential_pos_in_cross:
                for j in potential_pos_in_diag:
                  dist = np.linalg.norm(np.array(i) - np.array(j))
                  if dist == 1:
                      diag_values[[i for i in range(len(diag_indexes)) if diag_indexes[i] == j][0]] = 0
        
            # avoid connecting a neighbor that is part of another path when the previous point was defined as a node
            cross_values[[i for i in range(len(cross_indexes)) if cross_indexes[i] in to_avoid]] = 0
            diag_values[[i for i in range(len(diag_indexes)) if diag_indexes[i] in to_avoid]] = 0
            to_avoid = []
            
            # the number of paths defines whether or not a node needs to be created
            paths_number = sum([value == 1 for value in cross_values]+[value == 1 for value in diag_values])
            
            # handle a line that would finish in a already defined node (forming a closed loop)
            for position in cross_indexes + diag_indexes:
                if position in nodes:
                    if point in nodes_and_its_connections[position]:
                        if not (is_a_in_x([point, position], points) or is_a_in_x([position, point], points)):
                            points.append(position)
                            if paths_number != 0:
                                points.append(point)
                                nodes.append(point)
                                nodes_and_its_connections[point] = [cross_indexes[i] for i in range(len(cross_indexes)) if cross_values[i] ==1] + [diag_indexes[i] for i in range(len(diag_indexes)) if diag_values[i] ==1]
                        
        array[point] = 999 # point done
        array_as_list = list(array.reshape(array.size))
    
    # remove effect of the added border on points and nodes
    points = [(i[0]-1, i[1]-1) for i in points]
    nodes = [(i[0]-1, i[1]-1) for i in nodes]
    
    return points, nodes

def axes_swap(arg: list):
    reversedd = [tuple(reversed(element)) for element in arg]
    return reversedd

def edges_from_points_and_nodes(points: list, nodes: list):
    # create edges and stock them in a dict
    edges = {}
    reset_index = 0
    for index in range(len(points)):
        if points[index] in nodes:
            dist = np.linalg.norm(np.array(points[index]) - np.array(points[index-1]))
            if dist < 1.5 and points[index] != points[index-2]:
                edge = points[reset_index : index+1]
            else:
                edge = points[reset_index : index]

            if len(edge) > 1:
                edges['edge' + str(len(edges))] = edge
                
            reset_index = index
    # add last edge
    edge = points[reset_index : ]
    if len(edge) > 1:
        edges['edge' + str(len(edges))] = edge
        
    return edges
        
def graph_from_edges(edges: dict):    
    # retrieve nodes (this includes extremities) from edges
    nodes = []
    for key, value in edges.items():
        nodes.append(value[0])
        nodes.append(value[-1])
    
    # adding nodes to graph
    nodes = list(set(nodes))
    graph = nx.MultiGraph()
    for index in range(len(nodes)):
        graph.add_node(index, pos = tuple(nodes[index]))# rows and cols are swapped so that the graph uses the same axes than the initial array/image
    
    # adding edges to graph
    pos = nx.get_node_attributes(graph,'pos')
    for key, value in edges.items():
        first_node = [k for k, v in pos.items() if v == value[0]][0]
        last_node = [k for k, v in pos.items() if v == value[-1]][0]
        try:
            if first_node != last_node:
                line = LineString(value)
                graph.add_edge(first_node, last_node, weight= round(line.length))
        except:
            pass
        
    return graph

def window(array, i, j, d = 1):
    n = array[i-d:i+d+1, j-d:j+d+1].flatten()
    return n

def neighbors_index_and_value(array, row, col):
    cross_indexes = [(row-1, col), (row, col+1), (row+1, col), (row, col-1)]
    cross_values = np.array([array[index] for index in cross_indexes])
    diag_indexes = [(row-1, col+1), (row+1, col+1), (row+1, col-1), (row-1, col-1)]
    diag_values = np.array([array[index] for index in diag_indexes])
    return cross_indexes, cross_values, diag_indexes, diag_values

def is_a_in_x(a, x):
    for i in range(len(x) - len(a) + 1):
        if a == x[i:i+len(a)]: return True
    return False

def add_extremities_to_nodes(nodes: list, edges: dict):
    # add edge extremities as nodes
    for key, value in edges.items():
        if value[0] not in nodes:
            nodes.append(value[0])
        if value[-1] not in nodes:
            nodes.append(value[-1])
    
    return nodes