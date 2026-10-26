"""命令行评测入口（可复现）

用法：
    python -m evaluation.run_eval                 # 全量 100 题，关键词判分
    python -m evaluation.run_eval --limit 5       # 先跑 5 题冒烟
    python -m evaluation.run_eval --judge llm     # 用 LLM judge 判 correctness
    python -m evaluation.run_eval --retriever bm25 --out evaluation/results/manual

结果目录包含：
    results.jsonl  逐题明细（含检索/回答/指标/耗时）
    results.csv    便于 Excel 查看
    summary.json   汇总指标
    meta.json      模型、温度、检索后端等运行元信息
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.evaluation_runner import (  # noqa: E402
    evaluate_dataset,
    load_test_set,
    save_results,
    summarize,
)
from src.llm_client import LLMClient  # noqa: E402
from src.retrieval import build_retriever  # noqa: E402


def load_documents():
    docs = {}
    for device, path in config.MANUALS.items():
        with open(path, "r", encoding="utf-8") as f:
            docs[device] = f.read()
    return docs


def main():
    parser = argparse.ArgumentParser(description="SmartGuide 可复现评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    parser.add_argument("--judge", choices=["keyword", "llm"], default="keyword",
                        help="correctness 判分方式")
    parser.add_argument("--retriever", choices=["bigram", "bm25", "hybrid"],
                        default=None, help="覆盖 config.RETRIEVER")
    parser.add_argument("--sleep", type=float, default=0.5, help="每题间隔秒数")
    parser.add_argument("--out", default=None, help="结果输出目录")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rows = load_test_set()
    if args.limit:
        rows = rows[:args.limit]

    if args.retriever:
        config.RETRIEVER = args.retriever

    rag = build_retriever(load_documents())
    llm = LLMClient()
    evaluator = Evaluator()

    if not args.quiet:
        print(f"模型={config.LLM_MODEL} 检索={config.RETRIEVER} "
              f"题数={len(rows)} judge={args.judge}")

    def progress(idx, total, result):
        if args.quiet:
            return
        status = result.get("error") or result.get("category") or "-"
        grounded = result.get("groundedness")
        correct = result.get("correct")
        print(f"[{idx:>2}/{total}] {status} "
              f"grounded={grounded if grounded is None else round(grounded, 2)} "
              f"correct={correct} :: {result['question'][:30]}")

    results = evaluate_dataset(
        rows, rag, llm, evaluator,
        judge=args.judge, sleep_seconds=args.sleep, progress=progress,
    )
    summary = summarize(results)
    out_dir = save_results(
        results, summary, args.out,
        meta={"judge": args.judge, "n": len(rows)},
    )

    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {out_dir}")


if __name__ == "__main__":
    main()
