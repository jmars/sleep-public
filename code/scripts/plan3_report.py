#!/usr/bin/env python3
"""plan3 report: the pre-registered gates, the contrasts, and the stop-rule READ.

Every number here is RECOMPUTED from the per-question predictions in the arm summaries,
and each recomputation is checked against the stored pooled number it is supposed to
reproduce. That is deliberate (review2 B2's stale-artefact class, made structural): a
report that reads a stored verdict can publish a verdict the data no longer supports, and
the E5 sign defect was exactly a report reader disagreeing with the arms.

Nothing in this script decides anything. It computes the pre-registered gates and the
stop rule and PUBLISHES them; the decision belongs to the orchestrator and the user.

Pre-registered frames (plan3):
  F1 corpus_orig  every corpus question, EM primary
  F2 corpus_para  teacher paraphrase, gold = original answers; PRIMARY single-hop frame
                  if n >= 800 accepted, else F1 (decided from the build artefact before
                  any arm ran)
  F3 transfer     compositional two-hop, F1 primary (base near floor, EM granular)
  F4 heldout      never-trained passages

    python3 scripts/plan3_report.py --out runs/p6_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bootstrap_gate as bg  # noqa: E402

DEFAULT_ARMS = {"S": "p6_s", "C": "p6_c", "N": "p6_n"}
FRAMES = ("corpus_orig", "corpus_para", "transfer", "heldout")

# plan3 guards, stated here so the report and the record cannot drift apart
HELDOUT_MAX_DROP = 3.0
WITH_CONTEXT_MAX_DROP = 10.0
PAPER_CORPUS_GAP = 8.1          # 44.3 - 36.2, the paper's CORPUS column
STOP_NULL_ABS_PTS = 2.0
PARA_PRIMARY_MIN = 800


def load(run: str) -> dict:
    path = Path("runs") / run / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing arm summary: {path}")
    return json.loads(path.read_text())


def pairs(frame: dict, key: str) -> list[tuple[str, list[str]]]:
    preds = frame.get(key) or []
    golds = frame["golds"]
    return list(zip(preds, golds))


def pooled(frame: dict, key: str, metric: str) -> float:
    ps = pairs(frame, key)
    return 100.0 * sum(bg.score(p, g, metric) for p, g in ps) / max(len(ps), 1)


def check_alignment(arms: dict, frame: str) -> None:
    """Every arm must have read the SAME questions in the SAME order on this frame."""
    present = [a for a in arms if frame in arms[a]["summary"]["frames"]]
    if len(present) < 2:
        return
    first = arms[present[0]]["summary"]["frames"][frame]
    for name in present[1:]:
        other = arms[name]["summary"]["frames"][frame]
        if first["golds"] != other["golds"]:
            raise SystemExit(
                f"frame {frame} misaligned between {present[0]} and {name}: golds differ"
            )
        if len(first["predictions"]) != len(other["predictions"]):
            raise SystemExit(
                f"frame {frame} has {len(first['predictions'])} predictions in "
                f"{present[0]} and {len(other['predictions'])} in {name}"
            )


def recompute_checks(name: str, summary: dict, frames: dict) -> list[str]:
    """The stored pooled numbers must equal the recomputation from predictions."""
    problems = []
    # the stored-number key and the prediction key are NOT the same name: a report that
    # reads `final_predictions` finds nothing, scores 0.0 and "confirms" the wrong thing
    for frame, data in summary["frames"].items():
        for metric in ("em", "f1"):
            for side, key in (("base", "base_predictions"), ("final", "predictions")):
                stored = data[f"{side}_{metric}"]
                got = pooled(data, key, metric)
                if abs(stored - got) > 1e-6:
                    problems.append(
                        f"{name}/{frame}/{side}_{key}: stored {stored:.6f} != recomputed "
                        f"{got:.6f}"
                    )
    return problems


def contrast(arms: dict, frame: str, left: str, right: str, metric: str) -> dict:
    """Paired per-question bootstrap of `left - right` on one frame.

    The sign is asserted against the two arms' own pooled scores, which is the defect
    class the E5 report shipped: a reader that inverts its arguments publishes the
    opposite verdict, and a point estimate cannot tell you it has done so.
    """
    left_frame = arms[left]["summary"]["frames"][frame]
    right_frame = arms[right]["summary"]["frames"][frame]
    result = bg.paired(pairs(right_frame, "predictions"),
                       pairs(left_frame, "predictions"),
                       metric=metric, label=f"{left}-{right}@{frame}")
    expected = pooled(left_frame, "predictions", metric) - pooled(right_frame, "predictions", metric)
    if abs(result["gain_pts"] - expected) > 1e-6:
        raise SystemExit(
            f"sign/order defect in {left}-{right}@{frame} ({metric}): bootstrap says "
            f"{result['gain_pts']:+.4f} pts, arm scores say {expected:+.4f} pts"
        )
    return result


def own_base(arms: dict, name: str, frame: str, metric: str) -> dict:
    data = arms[name]["summary"]["frames"][frame]
    result = bg.paired(pairs(data, "base_predictions"), pairs(data, "predictions"),
                       metric=metric, label=f"{name}@{frame}")
    expected = pooled(data, "predictions", metric) - pooled(data, "base_predictions", metric)
    if abs(result["gain_pts"] - expected) > 1e-6:
        raise SystemExit(
            f"sign defect in {name} own-base @{frame} ({metric}): bootstrap says "
            f"{result['gain_pts']:+.4f}, arm scores say {expected:+.4f}"
        )
    return result


def transfer_strata(arms: dict, path: str = "data/transfer_twohop.jsonl") -> dict | None:
    """Split the transfer frame on the base model's own askability read.

    A transfer question the base can already answer from the SECOND passage alone is not
    a compositional question for this model, so the pooled transfer EM mixes two
    populations. `scripts/build_frames.py --askability` writes a PER-ROW reading
    (`askability.hop2_only`); the compositional stratum is the rows where that is 0. This
    is stratification telemetry, never an admission filter -- the frame is frozen and the
    pooled number is still reported.
    """
    frame_path = Path(path)
    if not frame_path.exists():
        return None
    rows = []
    with frame_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "question" in obj:      # the first line is the frame meta
                rows.append(obj)
    if not rows or "askability" not in rows[0]:
        return {"available": False,
                "reason": "frame carries no per-row askability read"}
    n = len(rows)
    for name, arm in arms.items():
        got = len(arm["summary"]["frames"]["transfer"]["predictions"])
        if got != n:
            return {"available": False,
                    "reason": f"frame has {n} rows, arm {name} has {got} predictions"}
    idx = [i for i, row in enumerate(rows) if row["askability"]["hop2_only"] == 0.0]
    out = {"available": True, "n_rows": n, "n_compositional": len(idx),
           "n_hop2_answerable": n - len(idx), "arms": {}}
    for name, arm in arms.items():
        frame = arm["summary"]["frames"]["transfer"]
        for metric in ("em", "f1"):
            base = [pairs(frame, "base_predictions")[i] for i in idx]
            final = [pairs(frame, "predictions")[i] for i in idx]
            out["arms"][f"{name}|{metric}"] = {
                "n": len(idx),
                "base": 100.0 * sum(bg.score(p, g, metric) for p, g in base) / max(len(base), 1),
                "final": 100.0 * sum(bg.score(p, g, metric) for p, g in final) / max(len(final), 1),
            }
    if len(idx) >= 10:
        left = [pairs(arms["S"]["summary"]["frames"]["transfer"], "predictions")[i] for i in idx]
        right = [pairs(arms["C"]["summary"]["frames"]["transfer"], "predictions")[i] for i in idx]
        out["S_minus_C"] = bg.paired(right, left, metric="f1",
                                     label="S-C@transfer_compositional")
    return out


def classify_ci(result: dict) -> str:
    lo, hi = result["ci95_pts"]
    if lo > 0:
        return "for_left"
    if hi < 0:
        return "against_left"
    return "spans_zero"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/p6_report.json")
    ap.add_argument("--run-s", default=DEFAULT_ARMS["S"])
    ap.add_argument("--run-c", default=DEFAULT_ARMS["C"])
    ap.add_argument("--run-n", default=DEFAULT_ARMS["N"])
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = {"S": args.run_s, "C": args.run_c, "N": args.run_n}
    arms = {name: {"run": run, "summary": load(run)} for name, run in runs.items()}

    problems: list[str] = []
    for name, arm in arms.items():
        problems.extend(recompute_checks(name, arm["summary"], arm["summary"]["frames"]))
    if problems:
        for line in problems:
            print(f"[RECOMPUTE MISMATCH] {line}")
        print(f"[FATAL] {len(problems)} stored numbers disagree with the predictions")
        return 1

    for frame in FRAMES:
        check_alignment(arms, frame)

    present = [f for f in FRAMES if all(f in a["summary"]["frames"] for a in arms.values())]
    n_para = len(arms["S"]["summary"]["frames"].get("corpus_para", {}).get("predictions", []))
    primary_single = "corpus_para" if n_para >= PARA_PRIMARY_MIN else "corpus_orig"
    print(f"[frames] available in all arms: {present}; para n={n_para} -> primary "
          f"single-hop frame = {primary_single}")

    boot: dict[str, dict] = {}
    for name in arms:
        for frame in present:
            boot[f"own_base|{name}|{frame}|em"] = own_base(arms, name, frame, "em")
            boot[f"own_base|{name}|{frame}|f1"] = own_base(arms, name, frame, "f1")

    for frame in present:
        for metric in ("em", "f1"):
            boot[f"S-C|{frame}|{metric}"] = contrast(arms, frame, "S", "C", metric)
            boot[f"S-N|{frame}|{metric}"] = contrast(arms, frame, "S", "N", metric)
            boot[f"C-N|{frame}|{metric}"] = contrast(arms, frame, "C", "N", metric)

    # ---- pre-registered conditions -----------------------------------------
    conditions: dict = {}
    single = {
        "frame": primary_single,
        "primary_metric": "em",
        "S_minus_C": boot[f"S-C|{primary_single}|em"],
        "S_minus_C_f1": boot[f"S-C|{primary_single}|f1"],
        "S_minus_N": boot[f"S-N|{primary_single}|em"],
        "S_gain_over_own_base": boot[f"own_base|S|{primary_single}|em"],
        "C_gain_over_own_base": boot[f"own_base|C|{primary_single}|em"],
        "N_gain_over_own_base": boot[f"own_base|N|{primary_single}|em"],
    }
    transfer = None
    if "transfer" in present:
        transfer = {
            "frame": "transfer",
            "primary_metric": "f1",
            "S_minus_C": boot["S-C|transfer|f1"],
            "S_minus_C_em": boot["S-C|transfer|em"],
            "S_minus_N": boot["S-N|transfer|f1"],
            "S_gain_over_own_base": boot["own_base|S|transfer|f1"],
            "C_gain_over_own_base": boot["own_base|C|transfer|f1"],
            "N_gain_over_own_base": boot["own_base|N|transfer|f1"],
        }

    strata = transfer_strata(arms) if "transfer" in present else None

    guards = {}
    for name, arm in arms.items():
        summary = arm["summary"]
        heldout = summary["frames"].get("heldout", {})
        heldout_drop = (heldout.get("base_em", 0.0) - heldout.get("final_em", 0.0)
                        if heldout else None)
        ctx = summary.get("with_context") or {}
        guards[name] = {
            "heldout_drop_pts": heldout_drop,
            "heldout_ok": None if heldout_drop is None else heldout_drop <= HELDOUT_MAX_DROP,
            "with_context_drop_pts": ctx.get("drop_pts"),
            "with_context_ok": (None if not ctx else ctx.get("drop_pts") <= WITH_CONTEXT_MAX_DROP),
            "literalism": summary.get("literalism"),
            "share_trained": (summary.get("share") or {}).get("trained_texts", {}).get("new_share"),
            "share_heldout": (summary.get("share") or {}).get("heldout_texts", {}).get("new_share"),
            "share_band": (summary.get("share") or {}).get("band"),
            "dose": summary.get("dose"),
            "arm_status": summary.get("arm_status"),
            "training": summary.get("training"),
        }

    # ---- the pre-registered STOP-CONTINUE read -----------------------------
    def frame_flags(entry: dict | None) -> dict | None:
        if entry is None:
            return None
        res = entry["S_minus_C"]
        return {
            "gain_pts": res["gain_pts"],
            "ci95_pts": res["ci95_pts"],
            "direction": classify_ci(res),
            "metric": entry["primary_metric"],
            "frame": entry["frame"],
        }

    single_flags = frame_flags(single)
    transfer_flags = frame_flags(transfer)
    flags = [f for f in (single_flags, transfer_flags) if f is not None]
    excl_for = [f for f in flags if f["direction"] == "for_left"]
    excl_against = [f for f in flags if f["direction"] == "against_left"]
    both_null = len(flags) == 2 and all(
        f["direction"] == "spans_zero" and abs(f["gain_pts"]) < STOP_NULL_ABS_PTS for f in flags
    )
    insufficient_frames = [
        f for f in flags if f["direction"] == "spans_zero"
        and f["gain_pts"] >= STOP_NULL_ABS_PTS
    ]
    s_guard_ok = bool(guards["S"]["heldout_ok"])
    c_guard_ok = bool(guards["C"]["heldout_ok"])

    if excl_against or (not s_guard_ok and c_guard_ok):
        stop_read = "STOP-NEGATIVE"
        stop_reason = ("CI95 excludes 0 against S on "
                       f"{[f['frame'] for f in excl_against]}" if excl_against else
                       "S fails non-destruction while C passes")
    elif excl_for and not excl_against:
        stop_read = "CONTINUE"
        stop_reason = f"CI95 excludes 0 in S favour on {[f['frame'] for f in excl_for]}"
    elif both_null:
        stop_read = "STOP-NULL"
        stop_reason = ("both frames CI95 contain 0 and |point| < 2.0 pts "
                       f"({[round(f['gain_pts'], 2) for f in flags]})")
    elif insufficient_frames:
        stop_read = "INSUFFICIENT"
        stop_reason = ("CI spans 0 with |point| >= 2.0 pts in S favour on "
                       f"{[f['frame'] for f in insufficient_frames]}; the ONE pre-registered "
                       "remediation is a seed-1 rerun of S and C only")
    else:
        stop_read = "UNCLASSIFIED"
        stop_reason = f"flags {flags} matched no pre-registered branch"

    # ---- secondary (informational only) ------------------------------------
    secondary = {
        "paper_corpus_gap_armed_pts": PAPER_CORPUS_GAP,
        "S_minus_N_on_primary": boot[f"S-N|{primary_single}|em"]["gain_pts"],
        "S_minus_N_ci95": boot[f"S-N|{primary_single}|em"]["ci95_pts"],
        "gap_reached": bool(
            boot[f"S-N|{primary_single}|em"]["gain_pts"] >= PAPER_CORPUS_GAP
            and boot[f"S-N|{primary_single}|em"]["ci95_pts"][0] > 0
        ),
        "note": "informational; NOT a continue/stop input",
    }

    # ---- dose-efficiency claim (para frame only) ---------------------------
    dose_efficiency = {"verdict": "UNMEASURED",
                       "reason": f"no corpus_para frame (n={n_para})"}
    if "corpus_para" in present:
        diff = bg.paired_gain_difference(
            pairs(arms["S"]["summary"]["frames"]["corpus_para"], "base_predictions"),
            pairs(arms["S"]["summary"]["frames"]["corpus_para"], "predictions"),
            pairs(arms["C"]["summary"]["frames"]["corpus_para"], "base_predictions"),
            pairs(arms["C"]["summary"]["frames"]["corpus_para"], "predictions"),
            metric="em", n_boot=args.n_boot, seed=args.seed, label="S_gain_minus_C_gain@para",
        )
        claimed = bool(diff["gain_diff_pts"] > 0 and diff["ci95_pts"][0] > 0)
        dose_efficiency = {
            "frame": "corpus_para",
            "metric": "em",
            "S_gain_minus_C_gain_pts": diff["gain_diff_pts"],
            "ci95_pts": diff["ci95_pts"],
            "verdict": "CLAIMED" if claimed else "UNPROVEN",
            "reason": ("paired CI excludes 0 in S favour" if claimed else
                       "paired CI does not exclude 0 in S favour; no third run"),
            "S_dose": guards["S"]["dose"], "C_dose": guards["C"]["dose"],
        }

    report = {
        "protocol": "plan3 full protocol (corpus, Table 3 CPT column shape)",
        "arms": {name: {"run": arm["run"],
                        "sft_mode": arm["summary"].get("sft_mode"),
                        "n_passages": arm["summary"].get("n_passages"),
                        "n_questions": arm["summary"].get("n_questions"),
                        "arm_status": arm["summary"].get("arm_status"),
                        "timings": arm["summary"].get("timings")}
                 for name, arm in arms.items()},
        "primary_single_hop_frame": primary_single,
        "n_para_accepted": n_para,
        "frames_present": present,
        "single_hop": single,
        "transfer": transfer,
        "transfer_strata": strata,
        "bootstraps": boot,
        "guards": guards,
        "secondary": secondary,
        "dose_efficiency": dose_efficiency,
        "stop_rule": {"read": stop_read, "reason": stop_reason,
                      "frames": flags,
                      "note": ("published as a READ; the decision is the orchestrator's "
                               "and the user's, not this script's")},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    # ---- printed table -----------------------------------------------------
    print(f"\n{'arm':4s} {'mode':7s} {'frame':12s} {'base':>7s} {'final':>7s} "
          f"{'gain':>7s} {'ci95':>18s}")
    for name in ("S", "C", "N"):
        for frame in present:
            res = boot[f"own_base|{name}|{frame}|em"]
            print(f"{name:4s} {arms[name]['summary'].get('sft_mode', '?'):7s} {frame:12s} "
                  f"{res['base_em']:7.2f} {res['final_em']:7.2f} {res['gain_pts']:+7.2f} "
                  f"[{res['ci95_pts'][0]:+7.2f},{res['ci95_pts'][1]:+7.2f}]")
    print(f"\n{'contrast':22s} {'metric':6s} {'pts':>8s} {'ci95':>20s} {'dir':>13s}")
    for key, res in boot.items():
        if key.startswith("own_base"):
            continue
        print(f"{key:22s} {res['metric']:6s} {res['gain_pts']:+8.2f} "
              f"[{res['ci95_pts'][0]:+8.2f},{res['ci95_pts'][1]:+8.2f}] "
              f"{classify_ci(res):>13s}")
    print(f"\n[stop rule] {stop_read}: {stop_reason}")
    print(f"[secondary, informational] S-N on {primary_single}: "
          f"{secondary['S_minus_N_on_primary']:+.2f} pts "
          f"CI {secondary['S_minus_N_ci95']} (armed {PAPER_CORPUS_GAP}) -> "
          f"gap_reached={secondary['gap_reached']}")
    print(f"[dose efficiency] {dose_efficiency['verdict']}: {dose_efficiency.get('reason')}")
    if strata and strata.get("available"):
        print(f"[transfer strata] {strata['n_compositional']}/{strata['n_rows']} rows are "
              f"not answerable from the second passage alone")
        for name in ("S", "C", "N"):
            em = strata["arms"][f"{name}|em"]
            print(f"  {name} compositional stratum EM {em['base']:.2f} -> {em['final']:.2f}")
        if "S_minus_C" in strata:
            res = strata["S_minus_C"]
            print(f"  S-C on the compositional stratum (f1): {res['gain_pts']:+.2f} "
                  f"CI [{res['ci95_pts'][0]:+.2f},{res['ci95_pts'][1]:+.2f}]")
    elif strata:
        print(f"[transfer strata] UNAVAILABLE: {strata.get('reason')}")
    print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
