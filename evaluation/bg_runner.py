"""后台评测进程。

由 Streamlit「一键评测」页面用 subprocess 拉起，脱离浏览器会话运行：
页面切走、刷新、关闭都不影响评测，进度和结果实时写到磁盘。

用法（页面会自动拼参数，也可手动跑）：
    python -m evaluation.bg_runner --judge keyword --retriever bm25 --sleep 0.5

产物（evaluation/results/<run_id>/）：
    progress.json  进度快照（页面轮询它）
    results.jsonl / results.csv / summary.json / meta.json  评测结束后由 save_results 写出
    eval.log       子进程 stdout/stderr
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

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


class StopRequested(Exception):
    """页面写了 STOP 标记文件，当前题跑完后收尾退出。"""


def write_progress(path, payload):
    """原子写：先写临时文件再 replace，避免页面读到半个 JSON。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="SmartGuide 后台批量评测")
    parser.add_argument("--judge", choices=["keyword", "llm"], default="keyword")
    parser.add_argument("--retriever", choices=["bigram", "bm25", "hybrid"], default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.EVAL_RESULT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    progress_path = os.path.join(run_dir, "progress.json")

    if args.retriever:
        config.RETRIEVER = args.retriever

    rows = load_test_set()
    if args.limit:
        rows = rows[:args.limit]

    state = {
        "run_id": run_id,
        "status": "running",
        "pid": os.getpid(),
        "judge": args.judge,
        "retriever": config.RETRIEVER,
        "sleep": args.sleep,
        "done": 0,
        "total": len(rows),
        "current": "",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": None,
        "error": None,
        "results": [],
    }
    write_progress(progress_path, state)
    print(f"[{run_id}] 开始评测 model={config.LLM_MODEL} "
          f"retriever={config.RETRIEVER} judge={args.judge} n={len(rows)}", flush=True)

    def flush():
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_progress(progress_path, state)

    try:
        rag = build_retriever(load_documents())
        llm = LLMClient()
        evaluator = Evaluator()

        stop_flag = os.path.join(run_dir, "STOP")

        def on_progress(idx, total, result):
            state["results"].append(result)
            state["done"] = idx
            state["current"] = result.get("question", "")[:40]
            flush()
            print(f"[{idx}/{total}] {result.get('category')} "
                  f"grounded={result.get('groundedness')} "
                  f"correct={result.get('correct')}", flush=True)
            if os.path.exists(stop_flag):
                raise StopRequested()

        results = evaluate_dataset(
            rows, rag, llm, evaluator, judge=args.judge,
            sleep_seconds=args.sleep, progress=on_progress,
        )
        summary = summarize(results)
        save_results(results, summary, out_dir=run_dir,
                     meta={"judge": args.judge, "n": len(rows),
                           "sleep": args.sleep, "run_id": run_id})
        state["status"] = "done"
        state["summary"] = summary
        state["current"] = ""
        flush()
        print(f"[{run_id}] 完成 -> {run_dir}", flush=True)
    except StopRequested:
        results = state["results"]
        summary = summarize(results) if results else None
        if summary:
            save_results(results, summary, out_dir=run_dir,
                         meta={"judge": args.judge, "n": len(results),
                               "stopped": True, "run_id": run_id})
        state["status"] = "stopped"
        state["summary"] = summary
        state["current"] = ""
        flush()
        print(f"[{run_id}] 已停止，保留 {len(results)} 条结果", flush=True)
    except Exception as e:  # noqa: BLE001 任何异常都要落盘，页面才能显示
        state["status"] = "failed"
        state["error"] = f"{type(e).__name__}: {e}"
        state["traceback"] = traceback.format_exc()
        flush()
        print(f"[{run_id}] 失败: {e}", flush=True)
        raise
    finally:
        time.sleep(0.2)


if __name__ == "__main__":
    main()
