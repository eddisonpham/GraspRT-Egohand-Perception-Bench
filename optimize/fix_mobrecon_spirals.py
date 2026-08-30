"""Hypothesis 1: Reconstruct MobRecon spirals from face winding in transform.pkl.

The original extract_spirals uses OpenMesh's mesh.vv() for half-edge-ordered
one-ring traversal. We can reconstruct this order from the face array by
following face winding: for face (v, a, b), a comes after v, b comes before v.
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TEMPLATE = Path.home() / "src" / "HandMesh" / "template"
p = pickle.load(open(TEMPLATE / "transform.pkl", "rb"), encoding="latin1")
faces_all = p["face"]
vertices_all = p["vertices"]

# Build vertex->face index for level 0 (778 verts, 1538 faces)
faces = faces_all[0]  # (1538, 3)
n_verts = len(vertices_all[0])  # 778

vert_faces = defaultdict(list)
for fi, tri in enumerate(faces):
    for pos, vi in enumerate(tri):
        vert_faces[int(vi)].append((fi, pos))


def ordered_one_ring(v, vert_faces_v):
    """Reconstruct half-edge winding order from face array."""
    if not vert_faces_v:
        return []
    # For each face containing v, record: the neighbor that follows v in winding
    # maps: after_v -> before_v (i.e., following face winding around v)
    after_to_before = {}
    for fi, pos in vert_faces_v:
        tri = faces[fi]
        after = int(tri[(pos + 1) % 3])
        before = int(tri[(pos - 1) % 3])
        after_to_before[after] = before
    if not after_to_before:
        return []
    # Walk the ring
    start = next(iter(after_to_before))
    ring = [start]
    current = start
    for _ in range(len(after_to_before) - 1):
        current = after_to_before[current]
        ring.append(current)
    return ring


# Get BFS neighbors from adj matrix for comparison
adj_dense = p["adj"][0].toarray() if hasattr(p["adj"][0], "toarray") else np.asarray(p["adj"][0])

print("=== Vertex one-ring comparison ===")
diffs = 0
same_set_diff_order = 0
for v in range(n_verts):
    ordered = ordered_one_ring(v, vert_faces[v])
    bfs = list(np.where(adj_dense[v])[0])
    if set(ordered) != set(bfs):
        diffs += 1
        if diffs <= 3:
            print(f"  DIFF SET v={v}: ordered={ordered[:8]} bfs={bfs[:8]}")
    elif ordered != bfs:
        same_set_diff_order += 1

print(f"\nTotal vertices: {n_verts}")
print(f"Different neighbor sets: {diffs}")
print(f"Same set, different order: {same_set_diff_order}")
print(f"Same set AND order: {n_verts - diffs - same_set_diff_order}")

# Now generate spirals from the ordered one-rings
from scipy.spatial import KDTree

def extract_spirals_ordered(seq_length, dilation=1):
    spirals = []
    for v in range(n_verts):
        ordered_ring = ordered_one_ring(v, vert_faces[v])
        spiral = [v]
        last_ring = ordered_ring
        # Simple ring expansion
        all_visited = set(spiral) | set(last_ring)
        spiral.extend(last_ring)
        while len(spiral) < seq_length * dilation and last_ring:
            next_ring = []
            for nv in last_ring:
                for nnv in ordered_one_ring(nv, vert_faces[nv]):
                    if nnv not in all_visited:
                        next_ring.append(nnv)
                        all_visited.add(nnv)
                        if len(spiral) + len(next_ring) >= seq_length * dilation:
                            break
                if len(spiral) + len(next_ring) >= seq_length * dilation:
                    break
            if not next_ring:
                break
            last_ring = next_ring
            spiral.extend(last_ring)
        # Fallback: KDTree if too short
        if len(spiral) < seq_length * dilation:
            pts = np.array(vertices_all[0])
            kdt = KDTree(pts, metric="euclidean")
            spiral = kdt.query(pts[v:v+1], k=seq_length * dilation, return_distance=False).tolist()[0]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return spirals

spirals = extract_spirals_ordered(seq_length=9, dilation=1)
spirals_arr = np.array(spirals)
print(f"\nSpirals shape: {spirals_arr.shape}")
print(f"Sample spiral[0]: {spirals[0][:12]}")

# Save for use by MobRecon wrapper
np.save(str(TEMPLATE / "spirals_ordered.npy"), spirals_arr)
print(f"Saved to {TEMPLATE / 'spirals_ordered.npy'}")
