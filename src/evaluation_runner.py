"""可复现评测：逐题执行、指标计算、结果落盘

与 Streamlit 页面解耦，命令行和 UI 共用同一套逻辑：
    python -m evaluation.run_eval --limit 5
    python -m evaluation.run_eval --judge llm --out evaluation/results/manual
"""
import csv
import json
import os
import re
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import config
from src.evaluator import Evaluator


def load_test_set(path: Optional[str] = None) -> List[Dict[str, str]]:
    """读取测试集，过滤重复表头行。"""
    path = path or config.TEST_SET_PATH
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # 兼容历史上手工维护时混入的重复表头行
    return [r for r in rows if (r.get("question") or "").strip()
            and r["question"].strip() != "question"]


def _judge_with_llm(llm_client, question: str, expected: str, answer: str) -> Dict:
    """用 LLM 判断回答是否覆盖参考答案的关键信息（0/1）。"""
    prompt = f"""你是客服问答评测员。请判断【模型回答】是否覆盖了【参考答案】的关键信息。

评分标准：
- 1 = 关键信息基本覆盖，用户按回答能解决问题（允许措辞不同）；
- 0 = 遗漏关键信息、答非所问，或包含与参考答案冲突的内容。

问题：{question}
参考答案：{expected}
模型回答：{answer}

只输出 JSON，不要解释：{{"score": 0或1, "reason": "一句话理由"}}"""
    try:
        raw = llm_client.complete(
            [{"role": "user", "content": prompt}], temperature=0.0
        )
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0)) if match else {}
        return {"score": int(data.get("score", 0)), "reason": data.get("reason", "")}
    except Exception as e:
        return {"score": None, "reason": f"judge 调用失败: {e}"}


def evaluate_one(
    row: Dict[str, str],
    rag,
    llm_client,
    evaluator: Evaluator,
    judge: str = "keyword",
) -> Dict:
    """评测单题，返回逐题明细。"""
    question = (row.get("question") or "").strip()
    expected = (row.get("expected_answer") or "").strip()
    gold_category = (row.get("category") or "").strip()

    result: Dict = {
        "question": question,
        "expected_answer": expected,
        "gold_category": gold_category,
        "answer": "",
        "category": "",
        "groundedness": None,
        "correctness_recall": None,
        "correct": None,
        "refused": None,
        "retrieved_count": 0,
        "latency_ms": None,
        "error": None,
    }

    started = time.time()
    try:
        retrieved = rag.retrieve(question, top_k=config.TOP_K)
        result["retrieved_count"] = len(retrieved)

        if not retrieved:
            result["answer"] = "该信息不在设备使用指南中，无法作答。"
            result["category"] = "超纲"
            result["refused"] = True
        else:
            answer = llm_client.answer_question(question, retrieved)
            result["answer"] = answer
            category, _ = evaluator.classify_question(question, answer)
            result["category"] = category
            result["refused"] = evaluator.is_refusal(answer)

            if category == "有效":
                grounded, _ = evaluator.calculate_accuracy_a(answer, retrieved)
                result["groundedness"] = grounded

            if expected:
                recall, detail = evaluator.calculate_correctness(answer, expected)
                result["correctness_recall"] = recall

                if gold_category == "out_of_scope":
                    # 超纲题的标准答案是「拒答」，只要正确拒答就算通过；
                    # 不拿关键词召回判分，否则拒答话术不同会被误判。
                    result["correct"] = bool(result["refused"])
                else:
                    result["correct"] = bool(detail.get("hit"))
                    if judge == "llm":
                        judged = _judge_with_llm(llm_client, question, expected, answer)
                        result["llm_judge"] = judged
                        if judged.get("score") is not None:
                            result["correct"] = bool(judged["score"])
    except Exception as e:
        result["category"] = "错误"
        result["error"] = str(e)
    finally:
        result["latency_ms"] = int((time.time() - started) * 1000)

    return result


def summarize(results: List[Dict]) -> Dict:
    """聚合指标。"""
    total = len(results)
    errors = [r for r in results if r.get("error")]
    out_of_scope = [r for r in results if r.get("gold_category") == "out_of_scope"]
    in_scope = [r for r in results if r.get("gold_category") != "out_of_scope"]

    grounded = [r["groundedness"] for r in results
                if r.get("groundedness") is not None]
    correctness = [r["correctness_recall"] for r in in_scope
                   if r.get("correctness_recall") is not None]
    correct_hits = [r for r in in_scope if r.get("correct") is True]
    refusal_hits = [r for r in out_of_scope if r.get("refused") is True]

    judged = [r for r in results if r.get("correct") is not None]
    success = [r for r in judged if r.get("correct") is True]

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "total": total,
        "valid": len([r for r in results if r.get("category") == "有效"]),
        "out_of_scope": len(out_of_scope),
        "invalid_or_error": len([r for r in results
                                 if r.get("category") in ("无效", "错误")]),
        "errors": len(errors),
        "groundedness_avg": avg(grounded),
        "groundedness_n": len(grounded),
        "correctness_recall_avg": avg(correctness),
        "correctness_n": len(correctness),
        "correct_rate": round(len(correct_hits) / len(in_scope), 4) if in_scope else None,
        "refusal_rate": round(len(refusal_hits) / len(out_of_scope), 4) if out_of_scope else None,
        "task_success_rate": round(len(success) / len(judged), 4) if judged else None,
        "task_success_n": len(judged),
    }


def save_results(results: List[Dict], summary: Dict, out_dir: Optional[str] = None,
                 meta: Optional[Dict] = None) -> str:
    """把逐题结果、汇总和运行元信息写入磁盘，返回目录。"""
    out_dir = out_dir or os.path.join(
        config.EVAL_RESULT_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_fields = ["question", "gold_category", "category", "groundedness",
                  "correctness_recall", "correct", "refused", "retrieved_count",
                  "latency_ms", "error", "answer"]
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": config.LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "retriever": config.RETRIEVER,
        "top_k": config.TOP_K,
        **(meta or {}),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    return out_dir


def evaluate_dataset(
    rows: List[Dict[str, str]],
    rag,
    llm_client,
    evaluator: Optional[Evaluator] = None,
    judge: str = "keyword",
    sleep_seconds: float = 0.0,
    progress: Optional[Callable[[int, int, Dict], None]] = None,
) -> List[Dict]:
    """顺序评测整个数据集。progress(idx, total, result) 供 UI 展示进度。"""
    evaluator = evaluator or Evaluator()
    results = []
    for idx, row in enumerate(rows, 1):
        result = evaluate_one(row, rag, llm_client, evaluator, judge=judge)
        results.append(result)
        if progress:
            progress(idx, len(rows), result)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return results
