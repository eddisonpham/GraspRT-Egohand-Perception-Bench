"""Candidate B — official MobRecon DenseStack adapter.

The upstream HandMesh repository is isolated at $MOBRECON_ROOT (default ~/src/HandMesh).
This uses the released mobrecon_densestack.pt checkpoint and the repo's exact CMR
preprocessing/spiral transforms. It intentionally does not use a detector: FreiHAND images
are already 224x224 hand crops, while detector-vs-reconstruction is separately represented
by WiLoR's end-to-end rows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.interface import BaseHandModel, HandPrediction  # noqa: E402


class MobReconHandModel(BaseHandModel):
    name = "mobrecon"

    def __init__(self, root: str | None = None):
        self.root = Path(root or os.environ.get("MOBRECON_ROOT", str(Path.home() / "src" / "HandMesh")))
        self.model = None
        self._device = "cpu"
        self.j_reg = None
        self.faces = None
        self._mean = 0.5
        self._std = 0.5

    def load(self, device: str = "cuda") -> None:
        import pickle
        import torch
        sys.path.insert(0, str(self.root))
        from cmr.models.mobrecon_densestack import MobRecon

        def load_transforms(transform_path, ds_factors, seq_lengths, dilations):
            # PROVISIONAL spiral ordering: transform.pkl ships precomputed, but the
            # released checkpoint was trained against OpenMesh's half-edge-ordered
            # one-ring traversal (extract_spirals in utils/generate_spiral_seq.py).
            # OpenMesh has no buildable wheel for py3.10 in this env and the conda
            # channel tops out at py3.8, so we approximate the spiral with BFS over
            # the face-derived adjacency. This does NOT reproduce the exact winding
            # order, so MobRecon accuracy is reported as PROVISIONAL and excluded
            # from the decision until OpenMesh is restored.
            with open(transform_path, "rb") as f:
                tmp = pickle.load(f, encoding="latin1")
            spirals = []
            for level in range(len(tmp["face"]) - 1):
                faces = np.asarray(tmp["face"][level])
                n = len(tmp["vertices"][level])
                adj = [[] for _ in range(n)]
                for tri in faces:
                    for aa, bb in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                        aa, bb = int(aa), int(bb)
                        if bb not in adj[aa]: adj[aa].append(bb)
                        if aa not in adj[bb]: adj[bb].append(aa)
                seq = int(seq_lengths[level] * dilations[level])
                level_spirals = []
                for root in range(n):
                    order, seen, frontier = [root], {root}, [root]
                    while frontier and len(order) < seq:
                        nxt = []
                        for aa in frontier:
                            for bb in adj[aa]:
                                if bb not in seen:
                                    seen.add(bb); order.append(bb); nxt.append(bb)
                                    if len(order) >= seq: break
                            if len(order) >= seq: break
                        frontier = nxt
                    if len(order) < seq: order.extend([root] * (seq - len(order)))
                    level_spirals.append(order[:seq:int(dilations[level])])
                spirals.append(torch.tensor(level_spirals, dtype=torch.long))
            def to_sparse(spmat):
                return torch.sparse_coo_tensor(
                    torch.from_numpy(np.vstack([spmat.tocoo().row, spmat.tocoo().col])).long(),
                    torch.from_numpy(spmat.tocoo().data).float(), torch.Size(spmat.shape))
            ups = [to_sparse(x) for x in tmp["up_transform"]]
            return spirals, ups, tmp
        dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        template = self.root / "template"
        ckpt = self.root / "downloads" / "mobrecon_densestack.pt"
        # fallback if checkpoint was copied into conventional CMR output path
        if not ckpt.exists():
            ckpt = self.root / "cmr" / "out" / "FreiHAND" / "mobrecon_spconv" / "checkpoints" / "mobrecon_densestack.pt"
        for p in [template / "template.ply", template / "transform.pkl", template / "j_reg.npy", template / "right_faces.npy", ckpt]:
            if not p.exists():
                raise FileNotFoundError(f"MobRecon asset missing: {p}")

        # Use the simple namespace expected by MobRecon, matching demo_mobrecon.sh.
        class A: pass
        a = A()
        a.out_channels = [32, 64, 128, 256]
        a.dsconv = False
        a.ds_factors = [2, 2, 2, 2]
        a.seq_length = [9, 9, 9, 9]
        a.dilation = [1, 1, 1, 1]
        a.size = 128
        a.in_channels = 3
        a.backbone = "DenseStack"
        # construct official graph transforms
        old = os.getcwd()
        os.chdir(self.root)
        try:
            spiral, up, tmp = load_transforms(str(template / "transform.pkl"), a.ds_factors, a.seq_length, a.dilation)
            # CMR model expects sparse tensor index/value tuples.
            up = [(*u._indices(), u._values()) for u in up]
            self.model = MobRecon(a, spiral, up)
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state, strict=True)
        finally:
            os.chdir(old)
        self.model = self.model.to(dev).eval()
        self._device = "cuda" if dev.type == "cuda" else "cpu"
        self._torch_device = dev
        self.j_reg = np.load(template / "j_reg.npy").astype(np.float32)
        self.faces = np.load(template / "right_faces.npy")
        self._size = 128

    def preprocess(self, image_bgr: np.ndarray):
        # CMR's base_transform expects RGB and performs resize + [0.5,0.5] normalization.
        from utils.vis import base_transform
        rgb = image_bgr[:, :, ::-1].copy()
        x = base_transform(rgb, size=self._size, mean=self._mean, std=self._std)
        import torch
        return torch.from_numpy(x).unsqueeze(0).to(self._torch_device)

    def infer(self, batch) -> list[HandPrediction]:
        import torch
        from cmr.datasets.FreiHAND.kinematics import mano_to_mpii
        with torch.no_grad():
            out = self.model(batch)
        mesh = out["mesh_pred"]
        if isinstance(mesh, list):
            mesh = mesh[0]
        mesh = mesh[0].detach().float().cpu().numpy()
        # Official CMR code unnormalizes mesh by std then maps MANO-order joints to MPII order.
        mesh = mesh * self._std
        joints = mano_to_mpii(self.j_reg @ mesh).astype(np.float32)
        joints = joints - joints[0:1]
        mesh = mesh - joints[0:1]
        uv = out.get("uv_pred")
        return [HandPrediction(
            joints_3d=joints,
            confidence=1.0,
            mano_pose=None,
            mano_shape=None,
            mesh_verts=mesh.astype(np.float32),
            handedness="right",
            raw={"uv_pred": uv.detach().cpu().numpy() if hasattr(uv, "detach") else None},
        )]

    @property
    def device(self) -> str:
        return self._device


if __name__ == "__main__":
    from data.freihand.loader import FreiHandLoader
    m = MobReconHandModel()
    m.load("cuda")
    L = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    img, _, _ = L[0]
    p = m.infer(m.preprocess(img))[0]
    print("joints:", p.joints_3d.shape, "verts:", p.mesh_verts.shape)
