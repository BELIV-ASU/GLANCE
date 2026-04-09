#!/usr/bin/env python3
"""
cf_gen.py  –  Road-Graph Counterfactual Data Generator
=======================================================
Implements Section III-A of the paper:

  "We first combine the road graph with instantaneous object locations to
   construct an object-associated road graph that binds each agent to its
   lane context. We then assign future textual meta-actions to each object
   and roll the scene forward, producing a temporal road graph that captures
   predicted object–lane occupancy over future timesteps."

Pipeline
--------
1.  Load nuPlan / DriveLM scene logs.
2.  For every key-frame, build a TemporalRoadGraph  G_t = (V_t, E_t).
3.  Enumerate a Cartesian product of ego × agent meta-action pairs.
4.  Roll each pair forward H steps and run the occupancy-overlap collision
    check described in the paper.
5.  Emit one JSONL line per (frame, action-pair) with:
        scene_token, frame_token, image_paths,
        ego_cf_action, agent_cf_actions,
        safety_label  ("safe" | "unsafe"),
        scene_understanding, crucial_objects   ← shared S1/S2 placeholders

Output
------
    <output_path>   –  JSONL consumed by data.py  --cf_json

Usage
-----
    python cf_gen.py \
        --data_json  /data/DriveLM/v1_0_train_nus.json \
        --map_root   /data/nuplan/maps \
        --output     /data/cf_train.jsonl \
        --horizon    3 \
        --dt         0.5 \
        --ds_res     1.0 \
        --max_scenes 0
"""

from __future__ import annotations

import argparse
import json
import math
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Meta-action vocabulary  (paper §III-A)
# ---------------------------------------------------------------------------

LAT_ACTIONS  = ["stay", "left", "right"]          # a_lat
LON_ACTIONS  = ["accelerate", "decelerate", "keep_speed"]  # a_lon

# Longitudinal acceleration magnitudes  (m/s²)
LON_ACCEL: Dict[str, float] = {
    "accelerate":   2.0,
    "decelerate":  -3.0,
    "keep_speed":   0.0,
}

# Human-readable labels written into JSONL (matches data.py / Fig. 3 style)
LAT_LABEL: Dict[str, str] = {
    "stay":  "keep lane",
    "left":  "change lane to left",
    "right": "change lane to right",
}
LON_LABEL: Dict[str, str] = {
    "accelerate":  "accelerate",
    "decelerate":  "decelerate",
    "keep_speed":  "keep speed",
}


def action_label(a_lat: str, a_lon: str) -> str:
    return f"{LAT_LABEL[a_lat]}, {LON_LABEL[a_lon]}"


# ---------------------------------------------------------------------------
# Road-graph data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """v = (lanelet_id, sample_index k, time τ)"""
    lanelet_id: str
    k: int           # arc-length sample index along the lanelet
    tau: int         # discrete time step

    def __hash__(self):
        return hash((self.lanelet_id, self.k, self.tau))

    def __eq__(self, other):
        return (self.lanelet_id, self.k, self.tau) == \
               (other.lanelet_id, other.k, other.tau)


@dataclass
class Lanelet:
    """Minimal HD-map lanelet representation."""
    lanelet_id: str
    centerline: List[Tuple[float, float]]   # (x, y) samples at ∆s_res spacing
    left_neighbor:  Optional[str] = None
    right_neighbor: Optional[str] = None
    successors:     List[str]     = field(default_factory=list)
    # Bounding polygon for occupancy check (list of (x,y) corners)
    polygon:        List[Tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def point_in_polygon(px: float, py: float,
                     polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def lanelet_polygon_at(lanelet: Lanelet, k: int,
                       ds_res: float, width: float = 3.5
                       ) -> List[Tuple[float, float]]:
    """
    Return a rectangular polygon centred on sample k of the lanelet
    centerline, approximating the spatial cell P_v used in the paper's
    occupancy check.

        width   – assumed lane width in metres (default 3.5 m)
    """
    pts = lanelet.centerline
    if not pts:
        return []
    k = max(0, min(k, len(pts) - 1))
    cx, cy = pts[k]

    # Tangent direction from neighbouring samples
    if k + 1 < len(pts):
        tx = pts[k + 1][0] - cx
        ty = pts[k + 1][1] - cy
    elif k > 0:
        tx = cx - pts[k - 1][0]
        ty = cy - pts[k - 1][1]
    else:
        tx, ty = 1.0, 0.0

    length = math.hypot(tx, ty) + 1e-9
    tx, ty = tx / length, ty / length
    nx, ny = -ty, tx                # normal (left)

    half_w = width / 2.0
    half_l = ds_res / 2.0

    return [
        (cx + tx * half_l + nx * half_w,  cy + ty * half_l + ny * half_w),
        (cx + tx * half_l - nx * half_w,  cy + ty * half_l - ny * half_w),
        (cx - tx * half_l - nx * half_w,  cy - ty * half_l - ny * half_w),
        (cx - tx * half_l + nx * half_w,  cy - ty * half_l + ny * half_w),
    ]


# ---------------------------------------------------------------------------
# Temporal Road Graph
# ---------------------------------------------------------------------------

class TemporalRoadGraph:
    """
    G_t = (V_t, E_t) as defined in §III-A.

    Nodes  v = (lanelet_id, k, τ)
    Edges  encode longitudinal, lateral, and topological transitions.
    """

    def __init__(self,
                 lanelets: Dict[str, Lanelet],
                 horizon:  int,
                 dt:       float,
                 ds_res:   float):
        self.lanelets = lanelets   # lanelet_id → Lanelet
        self.H        = horizon
        self.dt       = dt
        self.ds_res   = ds_res

    # ------------------------------------------------------------------
    # Frenet → nearest node projection  Π_τ(x_τ)
    # ------------------------------------------------------------------

    def project(self, lanelet_id: str, s: float, tau: int) -> Node:
        """Project arc-length position s to the nearest sample index k."""
        k = max(0, round(s / self.ds_res))
        ll = self.lanelets.get(lanelet_id)
        if ll:
            k = min(k, len(ll.centerline) - 1)
        return Node(lanelet_id, k, tau)

    # ------------------------------------------------------------------
    # Meta-action → successor node
    # ------------------------------------------------------------------

    def successor(self,
                  node:   Node,
                  a_lat:  str,
                  a_lon:  str,
                  speed:  float) -> Node:
        """
        Compute v_{τ+1} given current node and meta-action (§III-A).

            Δs    = μ_τ·Δt + ½·a_lon·Δt²
            n_lon = max(1, round(Δs / Δs_res))
        """
        a_lon_val = LON_ACCEL[a_lon]
        delta_s   = speed * self.dt + 0.5 * a_lon_val * self.dt ** 2
        n_lon     = max(1, round(abs(delta_s) / self.ds_res))

        # Lateral: choose target lanelet
        ll = self.lanelets.get(node.lanelet_id)
        if a_lat == "left" and ll and ll.left_neighbor:
            target_lane = ll.left_neighbor
        elif a_lat == "right" and ll and ll.right_neighbor:
            target_lane = ll.right_neighbor
        else:
            target_lane = node.lanelet_id

        # Advance along centerline; wrap to successor lanelet if needed
        new_k = node.k + n_lon
        tll   = self.lanelets.get(target_lane)
        if tll and new_k >= len(tll.centerline):
            # Topological continuation (iii)
            overflow = new_k - (len(tll.centerline) - 1)
            if tll.successors:
                target_lane = tll.successors[0]
                new_k       = min(overflow, len(self.lanelets[target_lane].centerline) - 1) \
                              if target_lane in self.lanelets else 0
            else:
                new_k = len(tll.centerline) - 1

        return Node(target_lane, new_k, node.tau + 1)

    # ------------------------------------------------------------------
    # Occupancy helpers
    # ------------------------------------------------------------------

    def node_polygon(self, node: Node) -> List[Tuple[float, float]]:
        ll = self.lanelets.get(node.lanelet_id)
        if ll is None:
            return []
        return lanelet_polygon_at(ll, node.k, self.ds_res)

    def agent_occupies(self,
                       bx: float, by: float,
                       node: Node) -> bool:
        """O[b, τ, v] = 1 iff agent BEV position (bx,by) ∈ P_v."""
        poly = self.node_polygon(node)
        if not poly:
            return False
        return point_in_polygon(bx, by, poly)

    # ------------------------------------------------------------------
    # Collision check  (paper §III-A)
    # ------------------------------------------------------------------

    def check_collision(self,
                        ego_node:    Node,
                        ego_a_lat:   str,
                        ego_a_lon:   str,
                        ego_speed:   float,
                        agents:      List[Dict]) -> bool:
        """
        Roll one step forward and check whether the ego-swept nodes overlap
        with any non-ego agent occupancy at τ+1.

        agents: list of dicts with keys:
                  lanelet_id, k, speed, a_lat, a_lon, bx, by
        Returns True (collision / unsafe) or False (safe).
        """
        # Ego swept nodes U_ego between τ and τ+1
        ego_next  = self.successor(ego_node, ego_a_lat, ego_a_lon, ego_speed)
        u_ego     = {ego_next}

        # Non-ego occupancy O¬ego_{τ+1}
        # For each agent: advance their node and check BEV overlap
        non_ego_nodes = set()
        for ag in agents:
            ag_node = Node(ag["lanelet_id"], ag["k"], ego_node.tau)
            ag_next = self.successor(ag_node, ag["a_lat"], ag["a_lon"], ag["speed"])
            # Check if agent BEV position falls inside any polygon at τ+1
            if self.agent_occupies(ag["bx"], ag["by"], ag_next):
                non_ego_nodes.add(ag_next)

        # Collision iff ∃ v : O_ego_{τ+1}(v) ∧ O¬ego_{τ+1}(v) = 1
        return bool(u_ego & non_ego_nodes)


# ---------------------------------------------------------------------------
# Stub HD-map loader  (replace with nuPlan MapAPI calls in production)
# ---------------------------------------------------------------------------

def load_lanelets_for_scene(scene_obj: Dict,
                            map_root:  Optional[Path]) -> Dict[str, Lanelet]:
    """
    Build a minimal lanelet dict from DriveLM scene data.

    In a full deployment wire this to the nuPlan MapFactory / HD-map SDK.
    Here we synthesise straight-line lanelets from whatever lane IDs appear
    in the frame metadata so the graph can run without the full map.
    """
    lanelets: Dict[str, Lanelet] = {}

    for frame_obj in (scene_obj.get("key_frames") or {}).values():
        ego_state = frame_obj.get("ego_state") or {}
        lane_id   = str(ego_state.get("lane_id", "lane_0"))

        if lane_id not in lanelets:
            # Synthesise a 50-sample straight centerline at 1 m spacing
            x0 = float(ego_state.get("x", 0.0))
            y0 = float(ego_state.get("y", 0.0))
            heading = float(ego_state.get("heading", 0.0))
            cx, cy  = math.cos(heading), math.sin(heading)
            centerline = [(x0 + cx * i, y0 + cy * i) for i in range(50)]
            lanelets[lane_id] = Lanelet(
                lanelet_id=lane_id,
                centerline=centerline,
            )

    # Fallback: always have at least one lane
    if not lanelets:
        lanelets["lane_0"] = Lanelet(
            lanelet_id="lane_0",
            centerline=[(float(i), 0.0) for i in range(50)],
        )

    return lanelets


# ---------------------------------------------------------------------------
# Agent state extractor
# ---------------------------------------------------------------------------

def extract_agents(frame_obj: Dict,
                   lanelets:  Dict[str, Lanelet],
                   ds_res:    float) -> Tuple[Dict, List[Dict]]:
    """
    Return (ego_state_dict, [agent_dict, ...]) from a DriveLM frame.

    ego_state_dict keys : lanelet_id, k, speed, bx, by
    agent_dict     keys : object_id, lanelet_id, k, speed, bx, by, a_lat, a_lon
                          (a_lat/a_lon are GT meta-actions for S3 supervision;
                           they are *replaced* during CF enumeration)
    """
    ego_raw   = frame_obj.get("ego_state") or {}
    ego_lane  = str(ego_raw.get("lane_id", "lane_0"))
    ego_s     = float(ego_raw.get("arc_length", 0.0))
    ego_speed = max(0.0, float(ego_raw.get("speed", 5.0)))
    ego_k     = max(0, round(ego_s / ds_res))
    ego_x     = float(ego_raw.get("x", 0.0))
    ego_y     = float(ego_raw.get("y", 0.0))

    ego_state = dict(
        lanelet_id=ego_lane,
        k=ego_k,
        speed=ego_speed,
        bx=ego_x,
        by=ego_y,
    )

    agents = []
    for obj_id, obj in (frame_obj.get("objects") or {}).items():
        a_lat_gt = obj.get("meta_lat", "stay")
        a_lon_gt = obj.get("meta_lon", "keep_speed")
        obj_lane = str(obj.get("lane_id", ego_lane))
        obj_s    = float(obj.get("arc_length", 0.0))
        obj_k    = max(0, round(obj_s / ds_res))
        agents.append(dict(
            object_id  = str(obj_id),
            lanelet_id = obj_lane,
            k          = obj_k,
            speed      = max(0.0, float(obj.get("speed", 5.0))),
            bx         = float(obj.get("x", 0.0)),
            by         = float(obj.get("y", 0.0)),
            a_lat      = a_lat_gt,
            a_lon      = a_lon_gt,
        ))

    return ego_state, agents


# ---------------------------------------------------------------------------
# Counterfactual enumeration for one frame
# ---------------------------------------------------------------------------

def enumerate_counterfactuals(
    scene_token:  str,
    frame_token:  str,
    frame_obj:    Dict,
    lanelets:     Dict[str, Lanelet],
    horizon:      int,
    dt:           float,
    ds_res:       float,
    # shared S1/S2 placeholders (filled by teacher VLM offline)
    scene_understanding: str = "",
    crucial_objects:     str = "",
) -> List[Dict]:
    """
    Build all (ego_action × agent_action) counterfactual rows for one frame.

    For each combination we roll H steps and flag safe / unsafe.
    We only keep rows where the label differs from the GT trajectory OR
    where the scenario is unsafe (to maximise unsafe coverage).
    """
    graph = TemporalRoadGraph(lanelets, horizon, dt, ds_res)
    ego_state, agents = extract_agents(frame_obj, lanelets, ds_res)

    if not agents:
        return []   # nothing to perturb

    image_paths = frame_obj.get("image_paths") or {}
    rows: List[Dict] = []

    # GT safety label from frame metadata (may be None for unlabelled data)
    gt_label = frame_obj.get("safety_label", None)

    # Enumerate ego × every agent's action pair
    ego_combos   = list(itertools.product(LAT_ACTIONS, LON_ACTIONS))
    agent_combos = list(itertools.product(LAT_ACTIONS, LON_ACTIONS))

    for (e_lat, e_lon) in ego_combos:
        ego_node = Node(ego_state["lanelet_id"], ego_state["k"], 0)

        for primary_agent in agents:
            for (a_lat, a_lon) in agent_combos:
                # Build agent list with primary agent's action perturbed;
                # all others keep GT actions
                cf_agents = []
                agent_cf_actions: Dict[str, str] = {}

                for ag in agents:
                    if ag["object_id"] == primary_agent["object_id"]:
                        cf_ag = dict(ag, a_lat=a_lat, a_lon=a_lon)
                        agent_cf_actions[ag["object_id"]] = action_label(a_lat, a_lon)
                    else:
                        cf_ag = dict(ag)
                        agent_cf_actions[ag["object_id"]] = action_label(
                            ag["a_lat"], ag["a_lon"]
                        )
                    cf_agents.append(cf_ag)

                # Roll forward H steps; flag collision at any step
                collision = False
                cur_node  = ego_node
                cur_speed = ego_state["speed"]
                for _ in range(horizon):
                    if graph.check_collision(
                        cur_node, e_lat, e_lon, cur_speed, cf_agents
                    ):
                        collision = True
                        break
                    cur_node   = graph.successor(cur_node, e_lat, e_lon, cur_speed)
                    cur_speed  = max(0.0, cur_speed + LON_ACCEL[e_lon] * dt)

                cf_label = "unsafe" if collision else "safe"

                # Skip exact GT replication (no perturbation effect)
                # but always keep unsafe-labelled rows for coverage
                if cf_label == gt_label and cf_label == "safe":
                    continue

                rows.append({
                    "scene_token":        scene_token,
                    "frame_token":        frame_token,
                    "image_paths":        image_paths,
                    "ego_cf_action":      action_label(e_lat, e_lon),
                    "agent_cf_actions":   agent_cf_actions,
                    "safety_label":       cf_label,
                    # Shared S1/S2 placeholders — filled offline by teacher VLM
                    "scene_understanding": scene_understanding,
                    "crucial_objects":     crucial_objects,
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate counterfactual JSONL from DriveLM + nuPlan data.\n"
            "Output is consumed by data.py --cf_json."
        )
    )
    parser.add_argument("--data_json",   required=True,
                        help="DriveLM scene JSON (v1_x_train_nus.json)")
    parser.add_argument("--map_root",    default="",
                        help="nuPlan HD-map root (optional; enables real lanelets)")
    parser.add_argument("--output",      required=True,
                        help="Output JSONL path  (e.g. cf_train.jsonl)")
    parser.add_argument("--horizon",     type=int,   default=3,
                        help="Roll-out horizon H (default 3 steps)")
    parser.add_argument("--dt",          type=float, default=0.5,
                        help="Time step Δt in seconds (default 0.5)")
    parser.add_argument("--ds_res",      type=float, default=1.0,
                        help="Arc-length sample resolution Δs_res in metres")
    parser.add_argument("--max_scenes",  type=int,   default=0,
                        help="Cap number of scenes processed (0 = all)")
    parser.add_argument("--unsafe_only", action="store_true",
                        help="Only emit unsafe counterfactual rows")
    args = parser.parse_args()

    map_root = Path(args.map_root) if args.map_root else None

    with open(args.data_json, "r", encoding="utf-8") as f:
        raw: Dict = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows   = 0
    total_unsafe = 0
    scene_count  = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for scene_token, scene_obj in raw.items():
            if args.max_scenes > 0 and scene_count >= args.max_scenes:
                break
            scene_count += 1

            lanelets = load_lanelets_for_scene(scene_obj, map_root)

            key_frames = scene_obj.get("key_frames") or {}
            for frame_token, frame_obj in key_frames.items():
                cf_rows = enumerate_counterfactuals(
                    scene_token  = scene_token,
                    frame_token  = frame_token,
                    frame_obj    = frame_obj,
                    lanelets     = lanelets,
                    horizon      = args.horizon,
                    dt           = args.dt,
                    ds_res       = args.ds_res,
                )

                for row in cf_rows:
                    if args.unsafe_only and row["safety_label"] != "unsafe":
                        continue
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows   += 1
                    total_unsafe += int(row["safety_label"] == "unsafe")

    n_safe = total_rows - total_unsafe
    print(f"Scenes processed : {scene_count}")
    print(f"CF rows written  : {total_rows:,}")
    print(f"  unsafe         : {total_unsafe:,}")
    print(f"  safe           : {n_safe:,}")
    print(f"Output           : {output_path}")


if __name__ == "__main__":
    main()