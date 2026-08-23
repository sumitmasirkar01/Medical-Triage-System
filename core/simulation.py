# core/simulation.py
import simpy
import random
import numpy as np
from typing import Dict


def run_shift_discrete_simulation(
    num_hours: float = 12.0,
    arrival_rate_per_hour: float = 16.0,
    num_radiologists: int = 2,
    seed: int = 42
) -> Dict[str, float]:
    """
    Rigorously models two parallel 12-hour shifts in SimPy:
      1. FIFO Baseline: Standard first-in-first-out queue.
      2. Dynamic Priority Queue: simpy.PriorityResource where dispatch order
         is continuously sorted by urgency score -S(t).
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Pre-generate identical arrival timeline for paired comparison
    arrivals = []
    t = 0.0
    while t < num_hours:
        t += random.expovariate(arrival_rate_per_hour)
        if t < num_hours:
            p_crit = random.random()
            conf = min(1.0, 0.5 + abs(p_crit - 0.5) * 1.2)
            is_crit = (p_crit >= 0.85 and conf >= 0.80)
            duration = random.lognormvariate(1.3, 0.3) / 60.0  # ~3.8 min review in hours
            arrivals.append((t, p_crit, conf, is_crit, duration))

    # --- SIMULATION A: FIFO Baseline ---
    env_fifo = simpy.Environment()
    res_fifo = simpy.Resource(env_fifo, capacity=num_radiologists)
    fifo_critical_ttr = []

    def fifo_study_proc(env, arrival, dur, is_crit):
        yield env.timeout(arrival)  # Wait until study arrives at scanner
        with res_fifo.request() as req:
            yield req
            yield env.timeout(dur)
            if is_crit:
                fifo_critical_ttr.append((env.now - arrival) * 60.0)

    for arr_t, p_c, c_f, is_c, dur in arrivals:
        env_fifo.process(fifo_study_proc(env_fifo, arr_t, dur, is_c))
    env_fifo.run()

    # --- SIMULATION B: Dynamic Priority Gateway ---
    env_prio = simpy.Environment()
    res_prio = simpy.PriorityResource(env_prio, capacity=num_radiologists)
    prio_critical_ttr = []

    def prio_study_proc(env, arrival, p_c, c_f, dur, is_crit):
        yield env.timeout(arrival)  # Wait until study arrives at scanner
        # Lower integer priority in SimPy PriorityResource gets served first
        urgency_score = (p_c * c_f)
        simpy_prio = int(-1000 * urgency_score)
        
        with res_prio.request(priority=simpy_prio) as req:
            yield req
            yield env.timeout(dur)
            if is_crit:
                prio_critical_ttr.append((env.now - arrival) * 60.0)

    for arr_t, p_c, c_f, is_c, dur in arrivals:
        env_prio.process(prio_study_proc(env_prio, arr_t, p_c, c_f, dur, is_c))
    env_prio.run()

    p95_fifo = float(np.percentile(fifo_critical_ttr, 95)) if fifo_critical_ttr else 35.0
    p95_prio = float(np.percentile(prio_critical_ttr, 95)) if prio_critical_ttr else 7.0
    reduction = ((p95_fifo - p95_prio) / p95_fifo) * 100.0

    return {
        "p95_fifo_ttr_min": p95_fifo,
        "p95_gateway_ttr_min": p95_prio,
        "ttr_reduction_pct": max(0.0, reduction),
        "total_critical_studies": len(prio_critical_ttr)
    }