"""
optimization_models.py
-----------------------
Deterministic MILP and two-stage stochastic programming (SP) baselines
for the multi-objective dispatch problem of eq. (7)-(11):

    min  F(x) = [f1(x) cost, f2(x) emissions, f3(x) supply risk]
    s.t. 0 <= E_s(t) <= E_s_max(t)                  for all sources s
         sum_s E_s(t) >= D(t)                       demand coverage
         E_HPP(t) + E_BCP(t) <= eta_B * B(t)         biofuel feasibility
         E_CPSS(t) <= E_CPSS_max(t)

The three objectives are scalarised via weighted sum (weights below);
f3 (supply risk) is approximated deterministically in the MILP as an
unmet-demand penalty, and handled properly via scenarios in the SP model.

Environment: Python 3.11, pulp (MILP/LP solver)
Seed = 42
"""
import numpy as np
import pandas as pd
import pulp

SEED = 42
np.random.seed(SEED)

SOURCES = ["E_SPP", "E_WPP", "E_HPP", "E_BCP", "E_CPSS"]

# unit cost (EUR/kWh) and emission coefficient (kg CO2/kWh) per source,
# consistent with the cost weights used in data generation
UNIT_COST = {"E_SPP": 0.021, "E_WPP": 0.026, "E_HPP": 0.052, "E_BCP": 0.061, "E_CPSS": 0.145}
EMISSION = {"E_SPP": 0.02, "E_WPP": 0.01, "E_HPP": 0.28, "E_BCP": 0.22, "E_CPSS": 0.45}

W_COST, W_EMIS, W_RISK = 0.6, 0.2, 0.2
ETA_B = 0.55
PENALTY_UNMET = 5.0  # EUR/kWh penalty for unmet demand (risk surrogate in MILP)


def solve_milp_month(D_t, B_t, E_max, E_cpss_max):
    """Deterministic single-period MILP dispatch (eq. 7-11 scalarised)."""
    prob = pulp.LpProblem("dispatch_milp", pulp.LpMinimize)

    E = {s: pulp.LpVariable(s, lowBound=0, upBound=E_max[s]) for s in SOURCES}
    E["E_CPSS"].upBound = E_cpss_max
    unmet = pulp.LpVariable("unmet", lowBound=0)

    cost = pulp.lpSum(UNIT_COST[s] * E[s] for s in SOURCES)
    emissions = pulp.lpSum(EMISSION[s] * E[s] for s in SOURCES)
    risk_penalty = PENALTY_UNMET * unmet

    prob += W_COST * cost + W_EMIS * emissions + W_RISK * risk_penalty

    prob += pulp.lpSum(E[s] for s in SOURCES) + unmet >= D_t          # eq. 11 (demand)
    prob += E["E_HPP"] + E["E_BCP"] <= ETA_B * B_t                     # eq. 11 (biofuel)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        **{s: E[s].value() for s in SOURCES},
        "unmet": unmet.value(),
        "cost": pulp.value(cost),
        "emissions": pulp.value(emissions),
        "objective": pulp.value(prob.objective),
        "status": pulp.LpStatus[prob.status],
    }


def solve_sp_month(D_t, B_scenarios, probs, E_max, E_cpss_max):
    """
    Two-stage stochastic programming dispatch: first-stage renewable/
    biofuel commitments are fixed before biofuel availability B is
    realised; second-stage grid draw (E_CPSS) recourses to cover any
    shortfall in each scenario, matching the risk term f3 of eq. (10).
    """
    prob = pulp.LpProblem("dispatch_sp", pulp.LpMinimize)
    n_s = len(B_scenarios)

    E_first = {s: pulp.LpVariable(s, lowBound=0, upBound=E_max[s])
               for s in ["E_SPP", "E_WPP", "E_HPP", "E_BCP"]}
    E_cpss = {k: pulp.LpVariable(f"E_CPSS_{k}", lowBound=0, upBound=E_cpss_max)
              for k in range(n_s)}
    unmet = {k: pulp.LpVariable(f"unmet_{k}", lowBound=0) for k in range(n_s)}

    first_cost = pulp.lpSum(UNIT_COST[s] * E_first[s] for s in E_first)
    first_emis = pulp.lpSum(EMISSION[s] * E_first[s] for s in E_first)

    recourse_cost = pulp.lpSum(
        probs[k] * (UNIT_COST["E_CPSS"] * E_cpss[k] + PENALTY_UNMET * unmet[k])
        for k in range(n_s)
    )
    recourse_emis = pulp.lpSum(probs[k] * EMISSION["E_CPSS"] * E_cpss[k] for k in range(n_s))

    prob += (W_COST * (first_cost + recourse_cost)
             + W_EMIS * (first_emis + recourse_emis))

    for k in range(n_s):
        prob += (E_first["E_SPP"] + E_first["E_WPP"] + E_first["E_HPP"]
                  + E_first["E_BCP"] + E_cpss[k] + unmet[k] >= D_t)
        prob += E_first["E_HPP"] + E_first["E_BCP"] <= ETA_B * B_scenarios[k]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        **{s: E_first[s].value() for s in E_first},
        "E_CPSS_expected": sum(probs[k] * E_cpss[k].value() for k in range(n_s)),
        "unmet_expected": sum(probs[k] * unmet[k].value() for k in range(n_s)),
        "objective": pulp.value(prob.objective),
        "status": pulp.LpStatus[prob.status],
    }


def main():
    df = pd.read_csv("data/energy_balance_2019_2023.csv")
    rng = np.random.default_rng(SEED)

    milp_rows, sp_rows = [], []
    for _, row in df.iterrows():
        E_max = {
            "E_SPP": 1.3 * row.E_SPP + 50, "E_WPP": 1.3 * row.E_WPP + 50,
            "E_HPP": 1.3 * row.E_HPP + 50, "E_BCP": 1.3 * row.E_BCP + 50,
            "E_CPSS": 1.5 * row.E_CPSS + 200,
        }
        milp_res = solve_milp_month(row.D, row.B, E_max, E_max["E_CPSS"])
        milp_rows.append({"date": row.date, **milp_res})

        # 20-scenario biofuel availability fan around the observed B
        B_scenarios = np.clip(row.B * (1 + rng.normal(0, 0.12, 20)), 0, None)
        probs = np.full(20, 1 / 20)
        sp_res = solve_sp_month(row.D, B_scenarios, probs, E_max, E_max["E_CPSS"])
        sp_rows.append({"date": row.date, **sp_res})

    pd.DataFrame(milp_rows).to_csv("results/milp_dispatch.csv", index=False)
    pd.DataFrame(sp_rows).to_csv("results/sp_dispatch.csv", index=False)
    print("Saved results/milp_dispatch.csv and results/sp_dispatch.csv")


if __name__ == "__main__":
    main()
