"""
Extract a network (networkx MultiGraph) from a skeletonized binary array.

Designed for skeletons produced by ``skimage.morphology.skeletonize`` (or any
equivalent thinning algorithm): a 2D array where 1s form a 1-pixel-wide set of
connected lines and 0s are background.

Main entry point
----------------
>>> from network_from_skeleton import network_from_skeleton
>>> G = network_from_skeleton(skeleton_array)

Nodes carry a ``pos = (x, y)`` attribute (matching the image when plotted)
and a ``kind`` attribute among:

* ``"end"``         — degree-1 node (line extremity),
* ``"junction"``    — degree >= 3 node (branching point),
* ``"subdivision"`` — degree-2 node kept only to encode exact pixel geometry
  (e.g. around diagonal crossings); removable with ``simplify="topology"``.

Edges carry a geometric ``weight`` (length of the pixel path, diagonal steps
counted as sqrt(2)) and a ``geometry`` attribute (list of pixels forming the
branch).
"""

from __future__ import annotations

import math
from collections import deque

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# 8-connectivity offsets: (di, dj)
_NEIGHBOURS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),          # cross (distance 1)
    (-1, -1), (-1, 1), (1, -1), (1, 1),        # diagonals (distance sqrt(2))
]


def _check_input(array: NDArray) -> None:
    """Validate that the input is a non-empty 2D binary array."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a numpy ndarray, got {type(array).__name__}")
    if array.ndim != 2:
        raise ValueError(f"array must be 2D, got {array.ndim} dimensions")
    if array.size == 0:
        raise ValueError("array is empty")
    unique = np.unique(array)
    if not np.isin(unique, [0, 1]).all():
        raise ValueError(f"array must be binary (0s and 1s only), found values {unique.tolist()}")


def _node_mask(skeleton: NDArray) -> NDArray:
    """
    Vectorized detection of nodes and dead-ends.

    A pixel is a "terminal" when its 8-connected neighbour count (restricted to
    skeleton pixels) is not equal to 2.
    """
    h, w = skeleton.shape
    padded = np.pad(skeleton.astype(np.uint8), 1)
    neighbour_sum = np.zeros((h, w), dtype=np.int16)
    for di, dj in _NEIGHBOURS:
        # pixel (i, j) of the skeleton maps to padded (i+1, j+1);
        # its neighbour (i+di, j+dj) maps to padded (i+1+di, j+1+dj).
        neighbour_sum += padded[1 + di : 1 + di + h, 1 + dj : 1 + dj + w]
    degree = np.where(skeleton > 0, neighbour_sum, 0)
    return (degree != 2).astype(np.uint8)


def _pop_component(start: tuple[int, int],
                   skeleton: NDArray,
                   visited: NDArray) -> list[tuple[int, int]]:
    """Collect, in BFS order, every skeleton pixel of one connected component."""
    h, w = skeleton.shape
    queue = deque([start])
    visited[start] = True
    component: list[tuple[int, int]] = []
    while queue:
        i, j = queue.popleft()
        component.append((i, j))
        for di, dj in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and skeleton[ni, nj] and not visited[ni, nj]:
                visited[ni, nj] = True
                queue.append((ni, nj))
    return component


def _neighbours_of(pixel: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    """All in-grid neighbours of a pixel (8-connectivity)."""
    i, j = pixel
    out = []
    for di, dj in _NEIGHBOURS:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            out.append((ni, nj))
    return out


def _step_candidates(pixel: tuple[int, int],
                     pixels: set[tuple[int, int]],
                     h: int, w: int) -> list[tuple[int, int]]:
    """
    Skeleton neighbours of a pixel that are valid "next steps".

    Diagonal pixels whose two adjacent cross-neighbours are also skeleton
    pixels represent an ambiguous corner: they are excluded to prevent jumping
    across an intersection.
    """
    i, j = pixel
    result = []
    for q in _neighbours_of(pixel, h, w):
        if q not in pixels:
            continue
        di, dj = q[0] - i, q[1] - j
        if abs(di) == 1 and abs(dj) == 1:
            if (i + di, j) in pixels and (i, j + dj) in pixels:
                continue  # ambiguous corner, handled via the cross step
        result.append(q)
    return result


def _split_into_paths(component: list[tuple[int, int]],
                      skeleton: NDArray) -> list[list[tuple[int, int]]]:
    """
    Decompose one connected component into maximal paths between terminals.

    Each path is a list of consecutive pixels going from one terminal pixel
    (dead-end or junction) to another. Pure cycles (no terminal at all) yield
    one cyclic path.
    """
    pixels = set(component)
    h, w = skeleton.shape
    visited_edges: set[frozenset] = set()

    def walk(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
        """Follow one branch from a terminal pixel until the next terminal."""
        path = [start, first]
        visited_edges.add(frozenset((start, first)))
        prev, cur = start, first
        while True:
            candidates = _step_candidates(cur, pixels, h, w)
            if len(candidates) > 2:
                break  # junction reached: its other branches get their own walks
            nxt = [q for q in candidates
                   if q != prev and frozenset((cur, q)) not in visited_edges]
            if not nxt:
                break  # dead end reached
            prev, cur = cur, nxt[0]
            path.append(cur)
            visited_edges.add(frozenset((prev, cur)))
        return path

    terminals = [p for p in component if len(_step_candidates(p, pixels, h, w)) != 2]

    paths: list[list[tuple[int, int]]] = []
    for t in terminals:
        for q in _step_candidates(t, pixels, h, w):
            if frozenset((t, q)) not in visited_edges:
                paths.append(walk(t, q))

    # Whatever remains is made of pure cycles (no terminal pixels).
    remaining = pixels - {p for path in paths for p in path}
    while remaining:
        start = min(remaining)
        for q in _step_candidates(start, pixels, h, w):
            if frozenset((start, q)) not in visited_edges:
                path = walk(start, q)
                if len(path) > 2:  # close the cycle if the walk came back
                    for q2 in _step_candidates(path[-1], pixels, h, w):
                        if q2 == start:
                            path.append(start)
                            visited_edges.add(frozenset((path[-2], start)))
                paths.append(path)
                remaining -= set(path)
                break
        else:
            remaining.discard(start)

    return paths


def _simplify_geometry(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop intermediate collinear pixels from a pixel path."""
    if len(path) <= 2:
        return list(path)
    keep = [path[0]]
    for k in range(1, len(path) - 1):
        di, dj = path[k][0] - keep[-1][0], path[k][1] - keep[-1][1]
        dn, dm = path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1]
        if (di, dj) != (dn, dm):
            keep.append(path[k])
    keep.append(path[-1])
    return keep


def _edges_from_paths(paths: list[list[tuple[int, int]]],
                      keep_geometry: bool) -> list[dict]:
    """Build edge records (start, end, geometry, length) from pixel paths."""
    records = []
    for path in paths:
        if len(path) < 2:
            continue
        geo = _simplify_geometry(path) if keep_geometry else [path[0], path[-1]]
        length = 0.0
        for a, b in zip(path, path[1:]):
            length += math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        records.append({"start": path[0], "end": path[-1],
                        "geometry": geo, "full_path": path,
                        "weight": round(length, 3)})
    return records


def _swap_axes_records(records: list[dict]) -> list[dict]:
    """Convert (row, col) pixel coordinates to (x, y) image coordinates."""
    for rec in records:
        rec["start"] = (rec["start"][1], rec["start"][0])
        rec["end"] = (rec["end"][1], rec["end"][0])
        rec["geometry"] = [(p[1], p[0]) for p in rec["geometry"]]
        rec["full_path"] = [(p[1], p[0]) for p in rec["full_path"]]
    return records


def _graph_from_edges(records: list[dict]) -> nx.MultiGraph:
    """Assemble the networkx MultiGraph from edge records."""
    graph = nx.MultiGraph()
    node_id: dict[tuple, int] = {}

    def get_node(pixel: tuple) -> int:
        if pixel not in node_id:
            node_id[pixel] = len(node_id)
            graph.add_node(node_id[pixel], pos=pixel)
        return node_id[pixel]

    for rec in records:
        u, v = get_node(rec["start"]), get_node(rec["end"])
        graph.add_edge(u, v, weight=rec["weight"], geometry=rec["geometry"])
    return graph


def _annotate_and_dissolve(graph: nx.MultiGraph,
                           dissolve_subdivisions: bool) -> nx.MultiGraph:
    def kinds(g: nx.MultiGraph) -> dict:
        return {n: ("end" if d == 1 else "junction" if d >= 3 else "subdivision")
                for n, d in g.degree()}

    nx.set_node_attributes(graph, kinds(graph), "kind")
    if not dissolve_subdivisions:
        return graph

    H = nx.MultiGraph(graph)
    merged = True
    while merged:
        merged = False
        for n in [x for x, d in H.degree() if d == 2]:
            u, v = list(H.neighbors(n))
            attrs_u = H.get_edge_data(u, n)
            attrs_v = H.get_edge_data(n, v)
            if not attrs_u or not attrs_v:
                continue  # securite: arrete rather than crash
            key_u = min(attrs_u, key=lambda k: attrs_u[k].get("weight", 1))
            key_v = min(attrs_v, key=lambda k: attrs_v[k].get("weight", 1))
            geo_u = attrs_u[key_u].get("geometry") or [H.nodes[u]["pos"], H.nodes[n]["pos"]]
            geo_v = attrs_v[key_v].get("geometry") or [H.nodes[n]["pos"], H.nodes[v]["pos"]]
            H.add_edge(u, v,
                       weight=attrs_u[key_u].get("weight", 1) + attrs_v[key_v].get("weight", 1),
                       geometry=geo_u[:-1] + geo_v)
            H.remove_node(n)
            merged = True
            break
    nx.set_node_attributes(H, kinds(H), "kind")
    return H


def network_from_skeleton(array: NDArray,
                          axesswap: bool = True,
                          keep_geometry: bool = True,
                          simplify: str = "geometry") -> nx.MultiGraph:
    """
    Return a weighted networkx MultiGraph built from a skeleton array.

    Parameters
    ----------
    array : ndarray, 2D
        Binary skeleton array (1s = skeleton pixels, 0s = background), e.g.
        the output of ``skimage.morphology.skeletonize``.
    axesswap : bool, default True
        If True, swap (row, col) to (x, y) so that node ``pos`` attributes
        match image coordinates when plotted with matplotlib.
    keep_geometry : bool, default True
        Store the pixel path of each branch in the ``geometry`` edge attribute.
    simplify : {"geometry", "topology", "none"}, default "geometry"
        * ``"geometry"``  — drop collinear intermediate pixels from stored
          geometries (weights always use the full path).
        * ``"topology"``  — additionally dissolve degree-2 subdivision nodes:
          every edge then connects an extremity or a junction only.
        * ``"none"``      — keep raw geometries (one pixel per path step).

    Returns
    -------
    networkx.MultiGraph
        Nodes carry ``pos = (x, y)`` and ``kind`` in
        {"end", "junction", "subdivision"}; edges carry ``weight`` (geometric
        length, diagonals = sqrt(2)) and ``geometry`` (pixel path).

    Raises
    ------
    TypeError, ValueError
        If the input is not a non-empty 2D binary numpy array.
    """
    _check_input(array)

    skeleton = array.astype(np.uint8)
    if not skeleton.any():
        return nx.MultiGraph()

    visited = np.zeros_like(skeleton, dtype=bool)
    records: list[dict] = []
    h, w = skeleton.shape

    for (i, j) in np.argwhere(skeleton > 0):
        if visited[i, j]:
            continue
        component = _pop_component((i, j), skeleton, visited)
        paths = _split_into_paths(component, skeleton)
        records.extend(
            _edges_from_paths(paths, keep_geometry=keep_geometry and simplify != "none")
        )

    if axesswap:
        records = _swap_axes_records(records)

    graph = _graph_from_edges(records)
    graph = _annotate_and_dissolve(graph, dissolve_subdivisions=(simplify == "topology"))
    return graph
