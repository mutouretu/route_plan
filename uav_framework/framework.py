# framework.py
# Reusable test harness for UAV spraying path experiments (TSP abstraction).
# - Load maps from CSV or generate randomly
# - Load pluggable methods from external modules (import string)
# - Apply refuel/reload constraints (battery + tank)
# - Compute metrics and export results

from __future__ import annotations
import math
import random
import importlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Callable
import pandas as pd
import numpy as np

Point = Tuple[float, float]

@dataclass
class UAVParams:
    v_spray_mps: float = 3.0
    v_transit_mps: float = 8.0
    battery_endurance_min: float = 18.0
    tank_capacity_l: float = 15.0
    flow_rate_lpm: float = 1.5
    turn_penalty_s: float = 2.0
    turnaround_time_min: float = 4.0

@dataclass
class MapSpec:
    # one of: {"type":"csv","path":"..."} or {"type":"random","N":100,"bounds":[xmin,ymin,xmax,ymax],"base_offset":30}
    type: str
    path: Optional[str] = None
    N: Optional[int] = None
    bounds: Optional[List[float]] = None
    base_offset: float = 30.0
    seed: int = 42

@dataclass
class ExperimentSpec:
    maps: List[MapSpec]
    methods: List[str]          # e.g. ["methods.examples:method1", "mypkg.mymod:my_method"]
    repeats: int = 1            # per (map, method) repetitions with different random seeds if applicable
    output_csv: str = "results.csv"
    uav_params: UAVParams = None

def import_method(method_path: str) -> Callable[[np.ndarray], List[int]]:
    # method_path format: "module.submodule:function_name"
    if ":" not in method_path:
        raise ValueError(f"method '{method_path}' must be 'module:function'")
    module_name, func_name = method_path.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    return fn

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dist_matrix(points: List[Point]) -> np.ndarray:
    n = len(points)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean(points[i], points[j])
            D[i, j] = D[j, i] = d
    return D

def load_map(spec: MapSpec) -> List[Point]:
    rng = random.Random(spec.seed)
    if spec.type == "csv":
        if not spec.path:
            raise ValueError("csv map requires 'path'")
        df = pd.read_csv(spec.path)
        base = df[df["type"]=="base"][["x","y"]].iloc[0].tolist()
        trees = df[df["type"]=="tree"][["x","y"]].values.tolist()
        points = [tuple(base)] + [tuple(t) for t in trees]
        return points
    elif spec.type == "random":
        if not spec.bounds or spec.N is None:
            raise ValueError("random map requires 'bounds' and 'N'")
        xmin, ymin, xmax, ymax = spec.bounds
        base = ((xmin + xmax)/2.0, ymin - spec.base_offset)
        trees = [(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(spec.N)]
        points = [base] + trees
        return points
    else:
        raise ValueError(f"unsupported map type: {spec.type}")

def segment_tour_into_sorties(tour: List[int], points: List[Point], params: UAVParams):
    """Split the closed tour into sorties respecting battery + tank constraints."""
    v_s = params.v_spray_mps
    v_t = params.v_transit_mps
    batt_s = params.battery_endurance_min * 60.0
    tank_s = (params.tank_capacity_l / params.flow_rate_lpm) * 60.0
    turn_pen = params.turn_penalty_s

    sorties = []
    seg = []
    t_spray_used = 0.0
    t_flight_used = 0.0

    def time_transit(i, j):
        return euclidean(points[i], points[j]) / v_t

    def time_spray(i, j):
        return euclidean(points[i], points[j]) / v_s

    for k in range(len(tour)-1):
        i, j = tour[k], tour[k+1]
        if (i == 0) or (j == 0):
            t_add = time_transit(i, j)
            if t_flight_used + t_add <= batt_s:
                seg.append((i, j, "transit"))
                t_flight_used += t_add
            else:
                sorties.append({"edges": seg, "t_flight_s": t_flight_used, "t_spray_s": t_spray_used})
                seg = [(i, j, "transit")]
                t_flight_used = t_add
                t_spray_used = 0.0
        else:
            t_add = time_spray(i, j) + turn_pen
            if (t_flight_used + t_add <= batt_s) and (t_spray_used + t_add <= tank_s):
                seg.append((i, j, "spray"))
                t_flight_used += t_add
                t_spray_used += t_add
            else:
                sorties.append({"edges": seg, "t_flight_s": t_flight_used, "t_spray_s": t_spray_used})
                seg = [(i, j, "spray")]
                t_flight_used = t_add
                t_spray_used  = t_add

    if seg:
        sorties.append({"edges": seg, "t_flight_s": t_flight_used, "t_spray_s": t_spray_used})

    return sorties

def compute_metrics(tour: List[int], points: List[Point], params: UAVParams) -> Dict[str, float]:
    # Ensure closed tour
    if tour[0] != 0: tour = [0] + tour
    if tour[-1] != 0: tour = tour + [0]

    sorties = segment_tour_into_sorties(tour, points, params)

    L_total = 0.0
    for k in range(len(tour)-1):
        a, b = tour[k], tour[k+1]
        L_total += euclidean(points[a], points[b])

    visited = set(tour)
    N = len(points)-1
    visited_trees = len([i for i in visited if i != 0])
    TCR = visited_trees / float(N) if N > 0 else 1.0

    T_op = 0.0
    for s in sorties:
        flight_min = s["t_flight_s"] / 60.0
        T_op += flight_min
    if len(sorties) >= 2:
        T_op += (len(sorties)-1) * params.turnaround_time_min

    return {
        "TCR": TCR,
        "L_total_m": L_total,
        "T_op_min": T_op,
        "num_sorties": len(sorties),
    }

def run_one(method: Callable[[np.ndarray], List[int]], points: List[Point], params: UAVParams, seed: int = 0) -> Dict[str, float]:
    D = dist_matrix(points)
    random.seed(seed); np_random = np.random.default_rng(seed)  # reserved for methods that use RNG
    tour = method(D)  # must return a list of node indices, starting/ending at 0 or not
    return compute_metrics(tour, points, params)

def run_from_config(cfg_path: str) -> pd.DataFrame:
    cfg = _load_config(cfg_path)
    rows = []
    for m in cfg["maps"]:
        map_spec = MapSpec(**m)
        points = load_map(map_spec)
        for method_path in cfg["methods"]:
            fn = import_method(method_path)
            for r in range(cfg.get("repeats", 1)):
                metrics = run_one(fn, points, UAVParams(**cfg.get("uav_params", {})), seed=cfg.get("seed", 42)+r)
                rows.append({
                    "map": m,
                    "method": method_path,
                    "N": len(points) - 1,
                    **metrics
                })
    df = pd.DataFrame(rows)
    out = cfg.get("output_csv", "results.csv")
    df.to_csv(out, index=False)
    return df

def _load_config(path: str) -> Dict:
    import json, yaml
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    elif path.endswith(".yml") or path.endswith(".yaml"):
        return yaml.safe_load(text)
    else:
        # try JSON first
        try:
            return json.loads(text)
        except Exception:
            import yaml as y
            return y.safe_load(text)
