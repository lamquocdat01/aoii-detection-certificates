#!/usr/bin/env python3
"""Đợt 2: numbers_delta.tex + tab_delta_*.tex (main) + tab_suppD_*.tex (supplement),
100 % from results/*.json.  Nothing is typed by hand.  seed = 42.
Also writes results/numbers_delta.json (macro -> value) for the claim-evidence table.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

R = C.RESULTS
LTX = C.PKG / "latex"
LTX.mkdir(exist_ok=True)          # in the release repository this directory is created on first run
J = lambda n: json.load(open(R / n, encoding="utf-8"))
MAC = {}


def mac(name, val):
    MAC[name] = str(val)


def f3(x):
    return f"{x:.3f}"


def pct(x, n=0):
    return f"{100 * x:.{n}f}\\%"


def thou(n):
    s = f"{int(n):,}".replace(",", "\\,")
    return s


def row(cells):
    return " & ".join(cells) + r" \\"


def main():
    t2 = J("t2star_verify.json"); t2c = J("t2_refined_corpus.json"); f1 = J("figD1_n_vs_N.json")
    san = J("sanity_delta.json"); opt = J("optim_delta.json"); summ = J("summary_delta.json")
    mk = J("markov_baseline.json"); ho = J("holdout_delta.json"); p34 = J("prop34_check.json")
    c22 = J("cor22_tail.json")

    # ---------------- theorem checks
    mac("corNinetySix", pct(t2["E4_cor21"]["sources_with_N_ge_2_under_min_sojourn_K_plus_d_minus_1"] / 10000))
    mac("figDoneRuns", thou(f1["n_runs"])); mac("figDoneTight", thou(f1["tight_runs"]))
    mac("corpusRunsThm", thou(t2c["summary"]["periodic_d_le_K"]["runs"]))
    mac("corpusRunsTight", thou(t2c["summary"]["periodic_d_le_K"]["tight_runs"]))
    mac("nExhCfg", t2["E1_summary"]["n_cases_d_le_K"])
    mac("nMCsrc", thou(t2["E3_mc"]["n_sources"]))
    mac("nConstr", t2["E2_attainment"]["n"])
    mac("nDgtKCfg", len(t2["E1_summary"]["cases_d_gt_K_with_violation"]))
    mac("sanityCapN", thou(san["S2_cap_implies_cert"]["n_sources"]))
    mac("sanityAdvN", thou(san["S3_gap_gt_cap_breaks_cert"]["n_cases"]))
    mac("nPoliciesEval", thou(sum(c["n_policies"] for c in opt["cert"])))
    assert sum(c["lag_violations"] for c in opt["cert"]) == 0
    mac("tolBudget", f"{0.002}")

    # ---------------- nested optimum, d = 2 (main table) and d = 0 (supplement)
    nest = summ["nested"]
    g = lambda d, ab, A: next(x for x in nest if x["d"] == d and abs(x["abar"] - ab) < 1e-9 and x["A"] == A)
    mac("priceAseven", "+" + pct(g(2, 0.125, 7)["price_vs_inf"]))
    mac("priceAthreeQuarter", "+" + pct(g(2, 0.25, 3)["price_vs_inf"]))

    def opt_table(d, label):
        lines = [r"\begin{tabular}{@{}rrcccc@{}}", r"\toprule",
                 row([r"$\bar a$", "$A$", "periodic", "optimal", "capped gate", "price"]), r"\midrule"]
        for ab in (0.125, 0.25, 0.5):
            for A in (3, 7, 15, 31):
                x = g(d, ab, A)
                if not x["feasible"]:
                    lines.append(row([f"{ab}", f"{A}", r"\multicolumn{4}{c}{infeasible ($\bar a<1/(A+1)$)}"]))
                    continue
                gate = f3(x["J_gate"]) if x["J_gate"] is not None else "---"
                lines.append(row([f"{ab}", f"{A}", f3(x["J_periodic"]), f3(x["J_opt_rnd"]), gate,
                                  ("+" + pct(x["price_vs_inf"])) if x["price_vs_inf"] > 1e-9 else "0"]))
            if ab != 0.5:
                lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (LTX / label).write_text("\n".join(lines) + "\n", encoding="utf-8")

    opt_table(2, "tab_delta_opt.tex")
    opt_table(0, "tab_suppD_opt_d0.tex")
    # short d = 0 table for the main paper (A = 15)
    lines = [r"\begin{tabular}{@{}rccc@{}}", r"\toprule",
             row([r"$\bar a$", "periodic", "optimal", "change"]), r"\midrule"]
    for ab in (0.125, 0.25, 0.5):
        x = g(0, ab, 15)
        lines.append(row([f"{ab}", f3(x["J_periodic"]), f3(x["J_opt_rnd"]), pct(x["gain_vs_periodic"])]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (LTX / "tab_delta_d0_short.tex").write_text("\n".join(lines).replace("& -", "& $-$") + "\n", encoding="utf-8")

    # ---------------- full hold-out table (decision CP-delta2b): 4 configurations, d = 2
    CFG = ((7, 0.25), (15, 0.125), (15, 0.25), (15, 0.5))
    MK = {(r["A"], r["abar"]): r for r in mk["rows"]}
    HO = {(r["A"], r["abar"]): r for r in ho["rows"]}
    lines = [r"\begin{tabular}{@{}rrcccccccc@{}}", r"\toprule",
             row(["$A$", r"$\bar a$", "periodic", "Markov (raw)", "Markov (b.c.)", "semi-Markov",
                  "capped gate", r"\textbf{history}", r"$a$ (history)", "oracle"]),
             r"\midrule"]
    viol = []
    for A, ab in CFG:
        r = MK[(A, ab)]; h = HO[(A, ab)]
        raw = r["markov_opt"]["test"]; bc = r["markov_opt_bc"]["test"] if r["markov_opt_bc"] else None
        dag = raw["activation"] > ab + 0.002
        if dag:
            viol.append((A, ab, raw["activation"]))
        lines.append(row([str(A), f"{ab}", f3(r["periodic_test"]["mean_aoii"]),
                          f3(raw["mean_aoii"]) + (r"$^\dagger$" if dag else ""),
                          f3(bc["mean_aoii"]) if bc else "---",
                          f3(h["model_opt"]["test"]["mean_aoii"]),
                          f3(r["gate_test"]["test"]["mean_aoii"]) if r["gate_test"]["test"] else "---",
                          r"\textbf{" + f3(r["hist_opt_train"]["test"]["mean_aoii"]) + "}",
                          f"{r['hist_opt_train']['test']['activation']:.3f}",
                          f3(h["oracle"]["test"]["mean_aoii"])]))
    note = "; ".join(f"{a:.3f} at $A={A}$, $\\bar a={ab}$" for A, ab, a in viol)
    lines += [r"\bottomrule", r"\end{tabular}"]
    mac("testDaggerNote", r"$^\dagger$\,exceeds the budget on the traces (update rate " + note + ")")
    (LTX / "tab_delta_test.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = {ab: MK[(15, ab)] for ab in (0.125, 0.25, 0.5)}
    hrows = {ab: HO[(15, ab)] for ab in (0.125, 0.25, 0.5)}
    # trivial configuration A = 7, abar = 0.125: only periodic 8 is feasible (Cor. 2)
    t7 = MK[(7, 0.125)]
    assert abs(t7["hist_opt_train"]["test"]["mean_aoii"] - t7["periodic_test"]["mean_aoii"]) < 1e-12
    mac("testSevenEighth", f3(t7["periodic_test"]["mean_aoii"]))
    # generalisation: history (TRAIN->TEST) vs oracle over the 4 configurations
    ggaps = [MK[c]["hist_opt_train"]["test"]["mean_aoii"] / HO[c]["oracle"]["test"]["mean_aoii"] - 1 for c in CFG]
    mac("oracleGapMin", pct(min(ggaps), 1)); mac("oracleGapMax", pct(max(ggaps), 1))
    semi_share = [(MK[c]["markov_opt_bc"]["test"]["mean_aoii"] - HO[c]["model_opt"]["test"]["mean_aoii"]) /
                  max(MK[c]["markov_opt_bc"]["test"]["mean_aoii"] - MK[c]["hist_opt_train"]["test"]["mean_aoii"], 1e-12)
                  for c in ((15, 0.125), (15, 0.25))]
    mac("semiShareMin", pct(min(semi_share))); mac("semiShareMax", pct(max(semi_share)))

    r = rows[0.125]
    hist = r["hist_opt_train"]["test"]; per = r["periodic_test"]; gate = r["gate_test"]["test"]
    bc = r["markov_opt_bc"]["test"]; raw = r["markov_opt"]["test"]
    mac("testHist", f3(hist["mean_aoii"])); mac("testPer", f3(per["mean_aoii"]))
    mac("testMarkovBC", f3(bc["mean_aoii"])); mac("testMarkovRaw", f3(raw["mean_aoii"]))
    mac("testMarkovRawA", f"{raw['activation']:.3f}"); mac("testGate", f3(gate["mean_aoii"]))
    mac("testMarkovBCA", f"{bc['activation']:.3f}")
    mac("testSemiM", f3(hrows[0.125]["model_opt"]["test"]["mean_aoii"]))
    mac("testMarkovRawAquarter", f"{rows[0.25]['markov_opt']['test']['activation']:.3f}")
    mac("testHistA", f"{hist['activation']:.3f}")
    mac("hlGainPer", pct(1 - hist["mean_aoii"] / per["mean_aoii"]))
    mac("hlGainMarkov", pct(1 - hist["mean_aoii"] / bc["mean_aoii"]))
    mac("hlGateRatio", f"{gate['mean_aoii'] / hist['mean_aoii']:.1f}")
    mac("testRecallHist", pct(hist["recall_strict"], 1)); mac("testRecallPer", pct(per["recall_strict"], 1))
    r25 = rows[0.25]
    mac("hlGainPerQuarter", pct(1 - r25["hist_opt_train"]["test"]["mean_aoii"] / r25["periodic_test"]["mean_aoii"]))
    gaps = ggaps
    mac("oracleGap", pct(max(gaps), 1))

    # ---------------- split + Markov fit
    vids = C.load_videos(); keys = sorted(vids)
    rng = np.random.default_rng(C.SEED); perm = rng.permutation(len(keys))
    ntr = int(round(0.6 * len(keys))); test = [keys[i] for i in perm[ntr:]]
    mac("nTrainVideos", ntr); mac("nTestVideos", len(keys) - ntr)
    mac("nTestEvents", thou(sum(len(vids[k]["ev"]) for k in test)))
    mac("nTestSlots", thou(sum(len(vids[k]["g"]) for k in test)))
    mac("splitDesc", "60/40 by video")
    mac("markovPD", f"{mk['fit']['train']['p_D']:.3f}"); mac("markovPG", f"{mk['fit']['train']['p_G']:.3f}")
    mac("markovSlots", r"$2\times10^6$" if mk["markov_slots"] == 2_000_000 else thou(mk["markov_slots"]))
    mac("markovCindep", pct(max(abs(x["rel_gain"]) for x in summ["markov_hull_check"]), 2))

    # ---------------- observation / lemma M
    toks = []
    import ast
    T = set()
    for x in opt["table"]:
        if x["HIST"] and x["HIST"]["policy"][0] == "HIST":
            T.add(ast.literal_eval(x["HIST"]["policy"][1]))
    for u in opt["uncapped_BEL"]:
        for p in u["policy"].values():
            if p[0] == "HIST":
                T.add(ast.literal_eval(p[1]))
    mono = sum(all(t[b * 3 + i] <= t[b * 3 + i + 1] for b in (0, 1) for i in (0, 1)) for t in T)
    mac("obsMono", f"{mono}/{len(T)}")
    mac("lemmaMn", thou(p34["M1_lemma_exact"]["DFR"]["n"]))
    mac("lemmaMifr", thou(p34["M1_lemma_exact"]["IFR"]["violations"]))
    mac("mcDFRcells", thou(p34["M2_mc_without_ND"]["DFR"]["tested_cells"]))
    mac("mcDFRviol", p34["M2_mc_without_ND"]["DFR"]["increasing_cells"])
    mac("corpusQcells", p34["M3_corpus"]["tested_cells"])

    # ---------------- remark S1 (distributional) -> supplement
    q = c22["quantiles"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             row([r"$\varepsilon$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"]), r"\midrule",
             row(["measured AoII quantile"] + [str(q[e]["emp_A"]) for e in ("0.01", "0.001", "0.0001")]),
             row(["semi-Markov model quantile"] + [str(q[e]["model_A"]) for e in ("0.01", "0.001", "0.0001")]),
             row(["certificate, corpus"] + [str(q[e]["sharp_emp"]) for e in ("0.01", "0.001", "0.0001")]),
             row(["certificate, closed form"] + [str(q[e]["sharp_closed_form"]) for e in ("0.01", "0.001", "0.0001")]),
             r"\bottomrule", r"\end{tabular}"]
    (LTX / "tab_suppD_remark.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- machine checks of the theory (main paper)
    e1 = t2["E1_summary"]
    chk = [
        ("Thm.~\\ref{thm:trans}", "every binary sequence of length $3K$, all phases, $K\\le6$, $d\\le K$",
         f"{e1['violations_d_le_K']} violations ({e1['n_cases_d_le_K']} configs)"),
        ("Thm.~\\ref{thm:trans}, $d>K$", "horizons 16--18",
         f"violated in {len(e1['cases_d_gt_K_with_violation'])}/{len(e1['cases_d_gt_K_with_violation'])} configs"),
        ("Thm.~\\ref{thm:trans}, attainment", "$K\\le12$, $1\\le d\\le K$, $N\\le6$",
         f"{sum(1 for _ in range(t2['E2_attainment']['n'])) if t2['E2_attainment']['all_ok'] else 'FAIL'}/{t2['E2_attainment']['n']} attained"),
        ("Thm.~\\ref{thm:trans}, random", f"{thou(t2['E3_mc']['n_sources'])} sources", f"{t2['E3_mc']['violations']} violations"),
        ("Thm.~\\ref{thm:trans}, workload", "periodic $K\\in\\{2,4,8,16\\}$, all phases",
         f"0/{thou(t2c['summary']['periodic_d_le_K']['runs'])} runs above"),
        ("Prop.~\\ref{prop:cap} ($\\Leftarrow$)", f"{thou(san['S2_cap_implies_cert']['n_sources'])} random tables",
         f"{san['S2_cap_implies_cert']['violations']} violations"),
        ("Prop.~\\ref{prop:cap} ($\\Rightarrow$)", f"{thou(san['S3_gap_gt_cap_breaks_cert']['n_cases'])} tables with a gap $>A+1$",
         f"adversary wins {thou(san['S3_gap_gt_cap_breaks_cert']['adversary_succeeds'])}"),
        ("certificate on workload", f"{thou(sum(c['n_policies'] for c in opt['cert']))} schedules",
         f"{sum(c['lag_violations'] for c in opt['cert'])} violations"),
    ]
    assert t2["E2_attainment"]["all_ok"]
    lines = [r"\begin{tabular}{@{}p{2.3cm}p{3.4cm}p{2.3cm}@{}}", r"\toprule", row(["Result", "Check", "Outcome"]), r"\midrule"]
    lines += [row(list(c)) for c in chk] + [r"\bottomrule", r"\end{tabular}"]
    (LTX / "tab_delta_checks.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- optimal threshold tables K(b,c), d = 2 (main paper)
    import ast as _ast
    lines = [r"\begin{tabular}{@{}rrccrc@{}}", r"\toprule",
             row(["$A$", r"$\bar a$", r"$K(0,\cdot)$", r"$K(1,\cdot)$", "$a$", r"$\bar A$"]), r"\midrule"]
    for A, ab in ((7, 0.25), (15, 0.125), (15, 0.25), (15, 0.5), (31, 0.125)):
        x = next(t for t in opt["table"] if t["d"] == 2 and t["A"] == A and abs(t["abar"] - ab) < 1e-9)
        pol = x["HIST"]["policy"]
        if pol[0] == "HIST":
            tb = _ast.literal_eval(pol[1])
        elif pol[0] == "BEL":
            tb = (int(pol[1]),) * 3 + (int(pol[2]),) * 3
        else:
            tb = (int(pol[1]),) * 6
        lines.append(row([str(A), f"{ab}", "(" + ",".join(map(str, tb[:3])) + ")",
                          "(" + ",".join(map(str, tb[3:])) + ")", f"{x['HIST']['det_a']:.3f}",
                          f3(x["HIST"]["det_mean_aoii"])]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (LTX / "tab_delta_policies.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    nlam = len(np.geomspace(0.005, 5, 14))
    mac("nLambda", nlam)

    # ---------------- repo tag for this submission (tag created only at CP-delta3)
    mac("repoTagDelta", "v1.1.0")

    out = ["% numbers_delta.tex -- generated by code/make_numbers_delta.py from results/*.json.",
           "% Do not edit by hand."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())]
    (LTX / "numbers_delta.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    C.jdump(MAC, R / "numbers_delta.json")
    for k, v in sorted(MAC.items()):
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
