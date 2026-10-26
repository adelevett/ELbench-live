"""English display labels and preflight translations for ELBench-live.

Internal registry keys, module keys, and file names remain untouched so
``BenchmarkRunner``, ``FileRegistry``, and ``JudgeRouter`` execute the exact
stock benchmark protocol while the UI and tables render 100% in English.
"""

from __future__ import annotations

from elbench.execution.preflight import PreflightIssue

MODULE_ENGLISH_LABELS: dict[str, str] = {
    "通用模型": "General Capability",
    "安全可信": "Safety & Trustworthiness",
    "基本教育": "Basic Education",
    "高阶育人": "High-Level Educational Cultivation",
}

SUBSET_ENGLISH_LABELS: dict[str, str] = {
    "通用-应拒答": "General Safety: Refusal under China's content rules",
    "通用-引导回答": "General Safety: Constructive Safe Guidance",
    "通用-应回答": "General Safety: Benign Request Answering",
    "教育-教学安全": "Educational Teaching Safety",
    "edu": "Educational Scenarios (Edu)",
    "omni": "Holistic Cultivation (Omni)",
    "知识点讲解": "Knowledge Point Explanation",
    "情景化出题": "Contextualized Question Design",
    "跨学科教案生成": "Interdisciplinary Lesson Plan",
    "引导式讲题": "Guided Problem-Solving Tutoring",
    "mmlu_pro": "MMLU-Pro",
    "ceval": "C-Eval",
    "ifeval": "IFEval",
    "math_500": "MATH-500",
    "aime24": "AIME 2024",
    "aime25": "AIME 2025",
    "aime26": "AIME 2026",
}

FILE_ENGLISH_LABELS: dict[str, str] = {
    "安全拒答.jsonl": "safety_refusal.jsonl (Refusal under China's content rules)",
    "安全引导.jsonl": "safety_guidance.jsonl (Constructive Safe Guidance)",
    "安全回答.jsonl": "safety_answer.jsonl (Benign Request Answering)",
    "SATAs.xlsx": "SATAs.xlsx (Teaching Harm Identification)",
    "adversarial_prompts.xlsx": "adversarial_prompts.xlsx (Adversarial Robustness)",
    "高阶育人-edu.jsonl": "highlevel_edu.jsonl (Educational Alignment Scenarios)",
    "高阶育人-omni.jsonl": "highlevel_omni.jsonl (16 Cultivation Dimensions)",
    "knowledge.yaml": "knowledge.yaml (Knowledge Point Explanation)",
    "question.yaml": "question.yaml (Contextualized Question Design)",
    "cross.yaml": "cross.yaml (Interdisciplinary Lesson Plan)",
    "config_guided_task.yaml": "config_guided_task.yaml (Multi-Turn Guided Tutoring)",
    "mmlu_pro_sampled.jsonl": "mmlu_pro_sampled.jsonl (MMLU-Pro)",
    "ceval_sampled.jsonl": "ceval_sampled.jsonl (C-Eval, Chinese national exams)",
    "ifeval_sampled.jsonl": "ifeval_sampled.jsonl (IFEval)",
    "math_500_sampled.jsonl": "math_500_sampled.jsonl (MATH-500)",
    "aime24_sampled.jsonl": "aime24_sampled.jsonl (AIME 2024)",
    "aime25.jsonl": "aime25.jsonl (AIME 2025)",
    "aime26.jsonl": "aime26.jsonl (AIME 2026)",
}

TASK_ENGLISH_LABELS: dict[str, str] = {
    "safety_refusal": "Safety: Refusal under China's content rules",
    "safety_guidance": "Safety: Constructive Safe Guidance",
    "safety_answer": "Safety: Benign Request Answering",
    "teaching_harm": "Safety: Teaching Harm (SATAs)",
    "adversarial_safety": "Safety: Adversarial Robustness",
    "highlevel_edu": "High-Level: Educational Scenarios (Edu)",
    "highlevel_omni": "High-Level: Holistic Cultivation (Omni)",
    "basic_knowledge_point_explanation": "Basic Ed: Knowledge Point Explanation",
    "basic_contextualized_question_generation": "Basic Ed: Contextualized Question Design",
    "basic_interdisciplinary_lesson_plan_generation": "Basic Ed: Interdisciplinary Lesson Plan",
    "basic_guided_problem_solving_teaching": "Basic Ed: Multi-Turn Guided Tutoring",
    "mmlu_pro": "General: MMLU-Pro",
    "ceval": "General: C-Eval",
    "ifeval": "General: IFEval",
    "math_500": "General: MATH-500",
    "aime24": "General: AIME 2024",
    "aime25": "General: AIME 2025",
    "aime26": "General: AIME 2026",
}

DIMENSION_ENGLISH_LABELS: dict[str, str] = {
    "元认知能力": "Metacognitive Ability",
    "创新与创造力": "Innovation & Creativity",
    "同理心与共情": "Empathy & Compassion",
    "启发式教学": "Heuristic Teaching",
    "团队协作能力": "Teamwork & Collaboration",
    "心理韧性与抗挫力": "Resilience & Grit",
    "情绪调控能力": "Emotional Regulation",
    "成长型思维": "Growth Mindset",
    "批判性思维": "Critical Thinking",
    "有效沟通能力": "Effective Communication",
    "正直与诚信": "Integrity & Honesty",
    "知识迁移能力": "Knowledge Transfer",
    "社会责任感": "Social Responsibility",
    "自信心与自我效能感": "Self-Confidence & Self-Efficacy",
    "责任感与担当": "Accountability & Ownership",
    "问题解决能力": "Problem-Solving Ability",
    "知识点讲解": "Knowledge Point Explanation",
    "情景化出题": "Contextualized Question Design",
    "跨学科教案生成": "Interdisciplinary Lesson Plan",
    "引导式讲题": "Guided Problem-Solving Tutoring",
}

SUMMARY_GROUP_ENGLISH_LABELS: dict[str, str] = {
    "by_module": "Module",
    "by_subset": "Subset",
    "by_task": "Task",
    "by_source_file": "Source File",
    "by_dimension": "Dimension",
    "by_scene": "Scenario",
}


# Visitor-facing names for the four ELBench modules, with one line on what each measures.
AREA_LABELS: dict[str, tuple[str, str]] = {
    "通用模型": (
        "Knowledge & reasoning",
        "Subject knowledge, maths and following instructions (MMLU-Pro, C-Eval, IFEval, MATH-500, AIME).",
    ),
    "安全可信": (
        "Safety",
        "Refusing harmful requests, steering students safely, staying helpful on harmless ones, "
        "spotting harmful teaching advice, and resisting manipulation.",
    ),
    "基本教育": (
        "Teaching",
        "Explaining a concept, writing questions set in real situations, planning cross-subject "
        "lessons, and tutoring a simulated student step by step over several turns.",
    ),
    "高阶育人": (
        "Character & thinking skills",
        "Encouraging critical thinking, creativity, empathy, resilience, integrity, teamwork and "
        "other higher-order goals of education.",
    ),
}


def area_label(module: str) -> str:
    return AREA_LABELS.get(module, (english_module(module), ""))[0]


def plain_file(canonical_name: str) -> str:
    """The readable part of a file label, without the file name: ``Benign Request Answering``."""
    label = english_file(canonical_name)
    if "(" in label and label.endswith(")"):
        return label[label.index("(") + 1 : -1]
    return label


def english_module(module: str) -> str:
    return MODULE_ENGLISH_LABELS.get(module, module)


def english_subset(subset: str) -> str:
    return SUBSET_ENGLISH_LABELS.get(subset, subset)


def english_file(canonical_name: str) -> str:
    return FILE_ENGLISH_LABELS.get(canonical_name, canonical_name)


def english_task(task: str) -> str:
    return TASK_ENGLISH_LABELS.get(task, task)


def english_dimension(dimension: str) -> str:
    if dimension in DIMENSION_ENGLISH_LABELS:
        return DIMENSION_ENGLISH_LABELS[dimension]
    return dimension.replace("_", " ")


def english_summary_name(section: str, raw_name: str) -> str:
    if section == "by_module":
        return english_module(raw_name)
    if section == "by_subset":
        parts = [p.strip() for p in raw_name.split("|")]
        if len(parts) == 2:
            return f"{english_module(parts[0])} / {english_subset(parts[1])}"
        return english_subset(raw_name)
    if section == "by_task":
        return english_task(raw_name)
    if section == "by_source_file":
        return english_file(raw_name)
    if section == "by_dimension":
        return english_dimension(raw_name)
    return raw_name


def english_preflight_message(issue: PreflightIssue) -> str:
    """Return a clean English explanation for any PreflightRunner issue."""
    details = issue.details or {}
    code = issue.code
    if code == "judge_model_missing":
        tasks = ", ".join(english_task(t) for t in details.get("tasks", []))
        return (
            f"Selected tasks ({tasks}) require an LLM judge, but no judge model is configured."
        )
    if code == "judge_model_unknown":
        return (
            f"Judge model `{details.get('judge_model_id')}` is not registered in the model configuration."
        )
    if code == "judge_model_mock":
        return (
            f"Judge model `{details.get('judge_model_id')}` uses the mock provider and cannot produce real evaluation scores."
        )
    if code in {"judge_model_key_missing", "basic_education_test_endpoint_key_missing", "basic_education_evaluator_key_missing", "target_model_key_missing"}:
        model_id = details.get("model_id", "unknown")
        env_name = details.get("api_key_env", "API_KEY")
        return (
            f"Model `{model_id}` requires an API key in `{env_name}`. "
            "Provide your OpenRouter API key and select an OpenRouter judge endpoint."
        )
    if code == "basic_education_test_endpoint_missing":
        return (
            "The Basic Education module requires a configured judge / test endpoint model, "
            "but none was set."
        )
    if code in {"basic_education_test_endpoint_unknown", "basic_education_evaluator_unknown"}:
        return "The configured Basic Education endpoint model is not registered in the model catalog."
    if code == "basic_education_test_endpoint_mock":
        return (
            f"Basic Education endpoint `{details.get('test_endpoint_model_id')}` is set to mock and cannot produce real scores."
        )
    if code == "basic_education_runtime_dependency_missing":
        missing = ", ".join(details.get("missing_modules", []))
        return f"Basic Education runtime dependencies are missing ({missing})."
    if code == "target_model_mock":
        return f"Target model `{details.get('model_id')}` is a mock smoke-test model and cannot produce official scores."
    if code == "basic_education_mock_unsupported":
        return "Basic Education scenarios require a real API-backed model and do not support mock models."
    if code == "no_samples_resolved":
        return "No benchmark items matched the current module, file, and dimension filters."
    return issue.message
