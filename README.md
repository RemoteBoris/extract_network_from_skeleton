# Note
Initially written at a time before generative AI, and by a geographer rather than a computer scientist, this code worked very well for what it was asked to do. Today, AI allows us to improve its robustness and readability — a nice illustration of how the tools at our disposal keep evolving.

# Extract network from skeleton
Module designed to extract a network (as networkx graph, see [MultiGraph doc](https://networkx.org/documentation/stable/reference/classes/multigraph.html) ) from a skeletonized array (such as the output of scikit-image skeletonize function for instance, see [Skeletonize doc](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html) ). 
Skeleton input being a binary array where zeros represent the background and ones represent a single-pixel wide skeleton.

## Framework
This module was initially created with the intention of solving the third step of a broader framework, e.i. extracting a network from a raster image (e.g. satellite image). The problem can be broken down into:
1. Distinguish network pixels from non-network pixels (whatever the algorithm, e.g. ML algorithms).
2. Skeletonize the binary image. 
3. Extract the network from the skeleton.

## Quick start
See demo [here](https://github.com/RemoteBoris/extract_network_from_skeleton/blob/main/network_from_skeleton_demo.ipynb).
