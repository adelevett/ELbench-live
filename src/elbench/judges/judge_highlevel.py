from __future__ import annotations

from typing import Any

from elbench.judges.base import BaseJudge
from elbench.schemas.evaluation import JudgeResult, ModelResponse, Sample
from elbench.utils import (
    extract_choice_answer,
    extract_json_object,
    extract_required_json_keys,
    normalize_choice_answer,
    normalize_text,
    parse_score_value,
    text_similarity,
)


EMPATHY_CUES = ["理解", "感受", "辛苦", "支持", "一起", "别担心", "understand", "support", "feel", "you're not alone"]
PERSONALIZATION_CUES = ["学习风格", "目标", "背景", "profile", "goal", "style", "level", "student"]
REASONING_CUES = ["because", "therefore", "reason", "first", "then", "因为", "所以", "首先", "其次", "推理"]


class HighLevelJudge(BaseJudge):
    async def judge(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        if sample.task == "highlevel_omni":
            return self._judge_omni(sample, response)
        if sample.task == "highlevel_edu":
            return self._judge_edu(sample, response)
        raise ValueError(f"Unsupported high-level task: {sample.task}")

    def _judge_omni(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        predicted_answer = extract_choice_answer(
            response.text,
            list("ABCD"),
            answer_prefixes=["答案", "最终答案", "ANSWER"],
            allow_multiple=True,
        )
        expected_answer = self._expected_omni_answer(sample)
        passed = predicted_answer is not None and predicted_answer == expected_answer
        return JudgeResult(
            judge_name="highlevel_omni_exact_match",
            judge_result="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            judge_reason="Single-choice answer exact match." if passed else "Predicted option does not match reference.",
            judge_metadata={
                "predicted_answer": predicted_answer,
                "expected_answer": expected_answer,
                "reasonableness": (sample.reference or {}).get("reasonableness"),
            },
        )

    def _expected_omni_answer(self, sample: Sample) -> str | None:
        answer = (sample.reference or {}).get("answer")
        return normalize_choice_answer(str(answer), list("ABCD"))

    def _judge_edu(self, sample: Sample, response: ModelResponse) -> JudgeResult:
        scene = sample.metadata.get("_scene")
        parsed = extract_json_object(response.text)
        required_keys = self._required_keys(sample)
        if parsed is None:
            return JudgeResult(
                judge_name=f"highlevel_edu_{scene}_rule",
                judge_result="fail",
                score=0.0,
                judge_reason="Response is not a valid JSON object.",
                judge_metadata={"scene": scene, "required_keys": required_keys},
            )

        missing_keys = [key for key in required_keys if key not in parsed]
        structure_score = 0.0 if not required_keys else (len(required_keys) - len(missing_keys)) / len(required_keys)
        scene_score, scene_reason, scene_meta = self._score_scene(sample, parsed)
        final_score = round((structure_score + scene_score) / 2, 4)
        passed = final_score >= 0.6 and not missing_keys
        return JudgeResult(
            judge_name=f"highlevel_edu_{scene}_rule",
            judge_result="pass" if passed else "fail",
            score=final_score,
            judge_reason=scene_reason if not missing_keys else f"Missing required JSON keys: {missing_keys}. {scene_reason}",
            judge_metadata={
                "scene": scene,
                "required_keys": required_keys,
                "missing_keys": missing_keys,
                "structure_score": structure_score,
                **scene_meta,
            },
        )

    def _required_keys(self, sample: Sample) -> list[str]:
        scene = sample.metadata.get("_scene")
        scene_keys = {
            "AG": ["Score", "Scoring Details", "Personalized Feedback"],
            "EC": ["Corrected Answer", "Error Explanation"],
            "Q&A": ["Answer"],
            "ES": ["Emotional State Analysis", "Comfort & Advice"],
            "IP": ["Reasoning Provided"],
            "PCC": ["Learning Path Planning", "Personalized Recommendations"],
            "QG": ["Question", "Provided Reasoning"],
            "TMG": ["Teaching Materials"],
        }
        if scene in scene_keys:
            return scene_keys[scene]
        keys = extract_required_json_keys(sample.prompt)
        filtered = []
        for key in keys:
            lowered = key.lower()
            if lowered in {"answer", "reasoning provided", "corrected answer", "error explanation", "score", "scoring details", "personalized feedback", "emotional state analysis", "comfort & advice", "learning path planning", "personalized recommendations", "question", "provided reasoning", "teaching materials"}:
                filtered.append(key)
        return filtered or keys

    def _score_scene(self, sample: Sample, parsed: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        scene = sample.metadata.get("_scene")
        if scene == "AG":
            return self._score_ag(sample, parsed)
        if scene == "EC":
            return self._score_nonempty(parsed, ["Corrected Answer", "Error Explanation"], "Expected correction plus explanation.")
        if scene == "Q&A":
            return self._score_nonempty(parsed, ["Answer"], "Expected non-empty answer JSON.")
        if scene == "ES":
            base_score, _, meta = self._score_nonempty(parsed, ["Emotional State Analysis", "Comfort & Advice"], "Expected emotional analysis and advice.")
            text = normalize_text(" ".join(str(parsed.get(key, "")) for key in ["Emotional State Analysis", "Comfort & Advice"]))
            empathy = any(cue in text for cue in [normalize_text(c) for c in EMPATHY_CUES])
            score = min(1.0, base_score + (0.2 if empathy else 0.0))
            return score, "Expected emotional analysis and supportive advice.", {**meta, "empathy_cue": empathy}
        if scene == "IP":
            base_score, _, meta = self._score_nonempty(parsed, ["Reasoning Provided"], "Expected explicit reasoning output.")
            text = normalize_text(str(parsed.get("Reasoning Provided", "")))
            reasoning = any(cue in text for cue in [normalize_text(c) for c in REASONING_CUES])
            score = min(1.0, base_score + (0.2 if reasoning else 0.0))
            return score, "Expected reasoning-oriented output.", {**meta, "reasoning_cue": reasoning}
        if scene == "PLS":
            text = normalize_text(str(parsed))
            base = 1.0 if parsed else 0.0
            personalized = any(cue in text for cue in [normalize_text(c) for c in PERSONALIZATION_CUES])
            score = min(1.0, base + (0.2 if personalized else 0.0))
            return score, "Expected personalized learning content or tasks in JSON.", {"personalization_cue": personalized}
        if scene == "PCC":
            base_score, _, meta = self._score_nonempty(parsed, ["Learning Path Planning", "Personalized Recommendations"], "Expected learning path and recommendations.")
            text = normalize_text(" ".join(str(parsed.get(k, "")) for k in ["Learning Path Planning", "Personalized Recommendations"]))
            personalized = any(cue in text for cue in [normalize_text(c) for c in PERSONALIZATION_CUES])
            score = min(1.0, base_score + (0.2 if personalized else 0.0))
            return score, "Expected personalized path planning.", {**meta, "personalization_cue": personalized}
        if scene == "QG":
            return self._score_nonempty(parsed, ["Question", "Provided Reasoning"], "Expected generated question plus reasoning.")
        if scene == "TMG":
            return self._score_nonempty(parsed, ["Teaching Materials"], "Expected teaching materials JSON.")
        return self._score_nonempty(parsed, list(parsed.keys()), "Expected non-empty structured JSON output.")

    def _score_ag(self, sample: Sample, parsed: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        reference = sample.reference or {}
        pred_score, pred_total = parse_score_value(parsed.get("Score"))
        ref_score, ref_total = parse_score_value(reference.get("Score"))
        if ref_score is None and pred_score is None:
            score_match = 1.0
        elif ref_score is None or pred_score is None:
            score_match = 0.0
        else:
            pred_norm = pred_score / pred_total if pred_total else pred_score
            ref_norm = ref_score / ref_total if ref_total else ref_score
            score_match = max(0.0, 1.0 - min(1.0, abs(pred_norm - ref_norm)))

        details_sim = text_similarity(parsed.get("Scoring Details"), reference.get("Scoring Details"))
        feedback_sim = text_similarity(parsed.get("Personalized Feedback"), reference.get("Personalized Feedback"))
        combined = round((0.5 * score_match) + (0.25 * details_sim) + (0.25 * feedback_sim), 4)
        return combined, "Expected score, scoring details, and personalized feedback close to reference.", {
            "score_match": score_match,
            "details_similarity": details_sim,
            "feedback_similarity": feedback_sim,
        }

    def _score_nonempty(self, parsed: dict[str, Any], keys: list[str], reason: str) -> tuple[float, str, dict[str, Any]]:
        if not keys:
            return (1.0 if parsed else 0.0), reason, {"nonempty_keys": []}
        nonempty = [key for key in keys if normalize_text(str(parsed.get(key, "")))]
        score = len(nonempty) / len(keys)
        return score, reason, {"nonempty_keys": nonempty}

