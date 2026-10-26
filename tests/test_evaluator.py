from src.evaluator import Evaluator


def test_groundedness_is_keyword_coverage():
    ev = Evaluator()
    docs = [("智能手环", "智能手环充满电后可续航 7 天，支持 5ATM 防水。", 0.9)]
    answer = "手环续航 7 天，支持 5ATM 防水。"
    score, detail = ev.calculate_accuracy_a(answer, docs)
    assert 0 <= score <= 1
    assert detail["total_keywords"] > 0


def test_correctness_recall_against_expected():
    ev = Evaluator()
    expected = "长按侧边电源键3秒开机"
    answer = "长按侧边电源键 3 秒即可开机。"
    recall, detail = ev.calculate_correctness(answer, expected)
    assert recall >= 0.6
    assert detail["hit"] is True


def test_correctness_misses_wrong_answer():
    ev = Evaluator()
    recall, detail = ev.calculate_correctness(
        "请咨询售卖商家。", "长按侧边电源键3秒开机"
    )
    assert recall < 0.6
    assert detail["hit"] is False


def test_groundedness_not_equal_correctness():
    """照抄文档但答非所问：groundedness 高，correctness 低。"""
    ev = Evaluator()
    docs = [("智能手环", "智能手环支持蓝牙连接，续航 7 天。", 0.9)]
    answer = "智能手环支持蓝牙连接，续航 7 天。"  # 没有覆盖标准答案
    g, _ = ev.calculate_accuracy_a(answer, docs)
    c, _ = ev.calculate_correctness(answer, "长按电源键 3 秒开机")
    assert g > c


def test_refusal_detection():
    ev = Evaluator()
    assert ev.is_refusal("该信息不在设备使用指南中，无法作答")
    assert not ev.is_refusal("长按电源键 3 秒开机")


def test_refusal_ignores_trailing_caveat():
    """答案主体正常、末尾补一句“无法作答”不算拒答。

    真实回归用例：run_100 里智能音箱场景模式被误判成超纲。
    """
    ev = Evaluator()
    answer = (
        "**在App中创建场景模式**，可实现一句话控制多个设备。\n\n"
        "文档中给出的场景示例：\n"
        "- **回家模式**：打开玄关灯、客厅灯，关闭窗帘，开启空调\n"
        "- **睡眠模式**：关闭所有灯，关闭电视，窗帘关闭\n\n"
        "如需了解App内具体操作步骤，该信息不在设备使用指南中，无法作答。"
    )
    assert ev.is_refusal(answer) is False
    category, _ = ev.classify_question("智能音箱如何创建场景模式？", answer)
    assert category == "有效"


def test_out_of_scope_template_still_counts_as_refusal():
    ev = Evaluator()
    assert ev.is_refusal(
        "该问题不在设备使用指南中，无法作答。请换个与设备使用相关的问法。"
    ) is True


def test_front_loaded_refusal_with_advice_still_counts():
    """开局就声明超纲、后面再补建议，仍算拒答（“哪个好”这类对比题）。"""
    ev = Evaluator()
    answer = (
        "该问题属于产品对比建议，不在设备使用指南范围内，无法作答。"
        "建议您根据自身需求（如屏幕尺寸、功能偏好、佩戴习惯等）综合考虑。"
    )
    assert ev.is_refusal(answer) is True
