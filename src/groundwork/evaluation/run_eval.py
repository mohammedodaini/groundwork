"""Benchmark runner.

    python -m groundwork.evaluation.run_eval --out docs/eval-results.md

IMPORTANT: this script writes ONLY numbers it measured. There is no default,
no placeholder and no example table baked in. If it has not been run against a
real provider, `docs/eval-results.md` does not exist, and the README says so.

Use `--provider fake` to smoke-test the harness itself with no API cost. Results
from the fake provider are labelled as such in the output and are meaningless as
quality measurements - they only prove the harness runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from groundwork.agent.graph import ResearchEngine
from groundwork.config import Settings, get_settings
from groundwork.domain.schemas import ResearchRequest
from groundwork.evaluation.scorers import TaskScore, score_result, summarise
from groundwork.logging_conf import configure_logging
from groundwork.providers.llm import build_llm
from groundwork.providers.search import ContentFetcher, build_search

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "benchmark.jsonl"


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            tasks.append(json.loads(line))
    return tasks


async def run_task(task: dict, settings: Settings, *, repeats: int = 1) -> list[TaskScore]:
    """Run one task `repeats` times.

    Repeats exist because LLM systems are stochastic even at temperature 0
    (batching, provider-side nondeterminism). A single run is an anecdote.
    """
    scores: list[TaskScore] = []
    for _ in range(repeats):
        engine = ResearchEngine(
            settings=settings,
            llm=build_llm(settings),
            search=build_search(settings),
            fetcher=ContentFetcher(settings),
        )
        request = ResearchRequest(
            objective=task["objective"],
            criteria=task.get("criteria", []),
            max_entities=task.get("max_entities", 3),
            max_search_rounds=task.get("max_search_rounds", 2),
            require_approval=False,
        )
        try:
            result = await engine.run(request)
            scores.append(
                score_result(
                    task["id"],
                    result,
                    expected_entities=task.get("expected_entities") or None,
                    forbidden_substrings=task.get("forbidden_substrings") or None,
                )
            )
        except Exception as exc:
            scores.append(TaskScore(task_id=task["id"], ok=False, error=str(exc)))
    return scores


def render_report(
    scores: list[TaskScore], settings: Settings, tasks: list[dict], repeats: int
) -> str:
    summary = summarise(scores)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Evaluation results",
        "",
        f"Generated: **{ts}**",
        f"Model: **{settings.llm_model}** via `{settings.llm_provider}`  ",
        f"Search: `{settings.search_provider}`  ",
        f"Tasks: **{len(tasks)}**, repeats per task: **{repeats}**",
        "",
    ]

    if not settings.uses_real_llm:
        lines += [
            "> **These numbers are from the `fake` provider.** They demonstrate that",
            "> the harness runs; they say nothing about research quality. Re-run with",
            "> a real provider before citing anything here.",
            "",
        ]

    lines += ["## Aggregate", "", summary.as_markdown(), "", "## Per task", ""]
    lines += [
        "| Task | OK | Entities | Claims | FACT | Evidence cov. | Unsupported | "
        "Quote verif. | Known-false | Latency (ms) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    by_task: dict[str, list[TaskScore]] = {}
    for s in scores:
        by_task.setdefault(s.task_id, []).append(s)

    for task_id, group in by_task.items():
        ok = [g for g in group if g.ok]
        if not ok:
            lines.append(f"| `{task_id}` | FAIL | - | - | - | - | - | - | - | - | - |")
            continue
        med = statistics.median
        lines.append(
            f"| `{task_id}` | {len(ok)}/{len(group)} | "
            f"{med([g.entities_found for g in ok]):.0f} | "
            f"{med([g.claims_total for g in ok]):.0f} | "
            f"{med([g.facts for g in ok]):.0f} | "
            f"{med([g.evidence_coverage for g in ok]):.0%} | "
            f"{med([g.unsupported_rate for g in ok]):.0%} | "
            f"{med([g.quote_verification_rate for g in ok]):.0%} | "
            f"{sum(g.forbidden_claims_present for g in ok)} | "
            f"{med([g.latency_ms for g in ok]):.0f} | "
            f"{sum(g.estimated_cost_usd for g in ok):.4f} |"
        )

    # Failure cases are part of the deliverable, not an embarrassment to hide.
    lines += ["", "## Observed failures and warnings", ""]
    any_issue = False
    for task_id, group in by_task.items():
        issues: list[str] = []
        for g in group:
            if not g.ok:
                issues.append(f"run errored: `{g.error}`")
            if g.forbidden_claims_present:
                issues.append(
                    f"produced {g.forbidden_claims_present} known-false claim(s)"
                )
            for w in g.warnings[:5]:
                issues.append(f"warning: {w}")
        if issues:
            any_issue = True
            lines.append(f"**`{task_id}`**")
            lines += [f"- {i}" for i in dict.fromkeys(issues)]
            lines.append("")
    if not any_issue:
        lines.append("_No failures or warnings recorded in this run._")

    lines += [
        "",
        "## How to read these metrics",
        "",
        "- **Evidence coverage** - share of FACT claims carrying at least one quote",
        "  that was verified verbatim against the fetched page. Deterministic.",
        "- **Unsupported rate** - share of claims the critic marked UNSUPPORTED,",
        "  CONTRADICTED or OVERSTATED. Partly LLM-judged, so treat as indicative.",
        "- **Quote verification rate** - share of all quotes the model produced that",
        "  actually appeared in the source. Deterministic, and the most direct",
        "  measure of fabrication.",
        "- **Known-false claims** - hand-written canary strings per task. Any number",
        "  above zero is a hallucination the system did not catch.",
        "",
        "There is deliberately no single 'accuracy' number. Research quality is not",
        "one-dimensional, and an LLM-judged accuracy score would be a model output",
        "reported as ground truth.",
    ]
    return "\n".join(lines)


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Groundwork benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=Path("docs/eval-results.md"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--task", type=str, default=None, help="Run a single task id.")
    parser.add_argument(
        "--provider", type=str, default=None, help="Override llm_provider."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.provider:
        settings = Settings(**{**settings.model_dump(), "llm_provider": args.provider})
    configure_logging(settings)

    tasks = load_tasks(args.dataset)
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"No task with id {args.task!r}", file=sys.stderr)
            return 2

    if not settings.uses_real_llm:
        print(
            "WARNING: running with the fake provider. Results are harness "
            "smoke-test output only, not a quality measurement.",
            file=sys.stderr,
        )

    all_scores: list[TaskScore] = []
    for task in tasks:
        print(f"-> {task['id']}", file=sys.stderr)
        all_scores.extend(await run_task(task, settings, repeats=args.repeats))

    report = render_report(all_scores, settings, tasks, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    raw = args.out.with_suffix(".json")
    raw.write_text(
        json.dumps([asdict(s) for s in all_scores], indent=2), encoding="utf-8"
    )

    print(f"\nWrote {args.out} and {raw}", file=sys.stderr)
    print(summarise(all_scores).as_markdown())
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
