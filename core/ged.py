import heapq
import networkx as nx
import torch
from torch_geometric.data import Data


def graph_to_nx(data: Data) -> nx.Graph:
    G = nx.Graph()
    n = data.num_nodes

    for i in range(n):
        feat = tuple(data.x[i].tolist()) if data.x is not None else ()
        G.add_node(i, feat=feat)

    edge_index = data.edge_index.cpu().numpy()
    seen = set()
    for col in range(edge_index.shape[1]):
        u, v = int(edge_index[0, col]), int(edge_index[1, col])
        if u != v and (u, v) not in seen and (v, u) not in seen:
            seen.add((u, v))
            G.add_edge(u, v)

    return G


def exact_ged(g1: Data, g2: Data, timeout: float = 10.0) -> float:
    G1 = graph_to_nx(g1)
    G2 = graph_to_nx(g2)

    def node_subst_cost(n1_attrs, n2_attrs):
        return 0.0 if n1_attrs.get("feat") == n2_attrs.get("feat") else 1.0

    def node_del_cost(attrs):
        return 1.0

    def node_ins_cost(attrs):
        return 1.0

    def edge_subst_cost(e1_attrs, e2_attrs):
        return 0.0

    def edge_del_cost(attrs):
        return 1.0

    def edge_ins_cost(attrs):
        return 1.0

    try:
        ged = nx.graph_edit_distance(
            G1, G2,
            node_subst_cost=node_subst_cost,
            node_del_cost=node_del_cost,
            node_ins_cost=node_ins_cost,
            edge_subst_cost=edge_subst_cost,
            edge_del_cost=edge_del_cost,
            edge_ins_cost=edge_ins_cost,
            timeout=timeout,
        )
        if ged is None:
            ged = _ged_upper_bound(G1, G2)
        return float(ged)

    except Exception:
        return _ged_upper_bound(G1, G2)


def _ged_upper_bound(G1: nx.Graph, G2: nx.Graph) -> float:
    return float(G1.number_of_nodes() + G1.number_of_edges()
                 + G2.number_of_nodes() + G2.number_of_edges())


def beam_search_ged(g1: Data, g2: Data, beam_width: int = 20) -> float:
    G1 = graph_to_nx(g1)
    G2 = graph_to_nx(g2)
    return _beam_ged_nx(G1, G2, beam_width)


def _node_cost(a1, a2):
    if a1 is None or a2 is None:
        return 1.0
    return 0.0 if a1.get("feat") == a2.get("feat") else 1.0


def _beam_ged_nx(G1: nx.Graph, G2: nx.Graph, beam_width: int) -> float:
    nodes1 = list(G1.nodes())
    nodes2 = list(G2.nodes())
    n1, n2 = len(nodes1), len(nodes2)

    if n1 == 0 and n2 == 0:
        return 0.0
    if n1 == 0:
        return float(n2 + G2.number_of_edges())
    if n2 == 0:
        return float(n1 + G1.number_of_edges())

    adj1 = nx.to_numpy_array(G1, nodelist=nodes1)
    adj2 = nx.to_numpy_array(G2, nodelist=nodes2)
    attrs1 = [G1.nodes[u] for u in nodes1]
    attrs2 = [G2.nodes[v] for v in nodes2]

    INF = float("inf")

    beam = [(0.0, 0, ())]

    best = INF

    for depth in range(n1):
        next_beam = []

        for cost, d, assignment in beam:
            if d != depth:
                continue
            u_idx = depth
            assigned_targets = set(j for j in assignment if j >= 0)

            for v_idx in range(n2):
                if v_idx in assigned_targets:
                    continue
                new_cost = cost + _node_cost(attrs1[u_idx], attrs2[v_idx])
            
                for prev_u, prev_v in enumerate(assignment):
                    if adj1[prev_u][u_idx] != adj2[prev_v][v_idx] if prev_v >= 0 else adj1[prev_u][u_idx]:
                        new_cost += 1.0
                new_assign = assignment + (v_idx,)
                heapq.heappush(next_beam, (new_cost, depth + 1, new_assign))

            del_cost = cost + 1.0
            for prev_u in range(depth):
                if adj1[prev_u][u_idx]:
                    del_cost += 1.0
            new_assign = assignment + (-1,)
            heapq.heappush(next_beam, (del_cost, depth + 1, new_assign))

        if not next_beam:
            break

        next_beam = heapq.nsmallest(beam_width, next_beam)
        beam = next_beam

    for cost, d, assignment in beam:
        if d != n1:
            continue
        assigned_targets = set(j for j in assignment if j >= 0)
        unmatched_g2 = [j for j in range(n2) if j not in assigned_targets]

        extra = len(unmatched_g2)

        for a in range(len(unmatched_g2)):
            for b in range(a + 1, len(unmatched_g2)):
                if adj2[unmatched_g2[a]][unmatched_g2[b]]:
                    extra += 1

        total = cost + extra
        if total < best:
            best = total

    return float(best) if best < INF else _ged_upper_bound(G1, G2)