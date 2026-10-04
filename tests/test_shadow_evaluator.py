from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "evaluation"))

from evaluate_shadow import (  # noqa: E402
    EvaluationInputError,
    GoldenSet,
    ShadowLog,
    YOLO_CLASSES,
    evaluate,
    load_golden,
    load_shadow,
)


FILE_INPUT = {
    "kind": "file",
    "sha256": "a" * 64,
    "video_stride": 1,
}
CAPTURE_INPUT = {
    "kind": "capture",
    "capture_id": "capture-20261003-001",
    "video_stride": 1,
}
STAGE2_CORE_FIELDS = (
    "status",
    "candidate_class",
    "predicted_class",
    "score",
    "score_kind",
    "second_best_class",
    "second_best_score",
    "margin",
    "class_scores",
)
TEMPORAL_RAW_FIELDS = (
    "status",
    "predicted_class",
    "candidate_class",
    "score",
    "second_best_class",
    "second_best_score",
    "margin",
)


def yolo_scores(selected: str) -> dict[str, float]:
    scores = {label: 0.01 for label in YOLO_CLASSES}
    scores[selected] = 0.9
    return scores


def stage2_scores(major: str, candidate: str | None) -> dict[str, float]:
    if candidate is None:
        return {}
    scores = {f"{major}{index}": 0.01 * index for index in range(1, 11)}
    scores[candidate] = 0.9
    return scores


def stage2_evidence(
    major: str,
    candidate: str | None,
    predicted: str | None,
    status: str,
) -> dict[str, object]:
    scores = stage2_scores(major, candidate)
    if not scores:
        return {
            "status": status,
            "candidate_class": None,
            "predicted_class": predicted,
            "score": None,
            "score_kind": None,
            "second_best_class": None,
            "second_best_score": None,
            "margin": None,
            "class_scores": {},
        }
    assert candidate is not None
    competitors = sorted(
        (label for label in scores if label != candidate),
        key=scores.__getitem__,
        reverse=True,
    )
    second = competitors[0]
    return {
        "status": status,
        "candidate_class": candidate,
        "predicted_class": predicted,
        "score": scores[candidate],
        "score_kind": (
            "cosine_similarity" if major == "f" else "softmax_confidence"
        ),
        "second_best_class": second,
        "second_best_score": scores[second],
        "margin": scores[candidate] - scores[second],
        "class_scores": scores,
    }


def add_temporal_evidence(
    raw: dict[str, object], decision: dict[str, object]
) -> None:
    status = decision["status"]
    candidate = decision["candidate_class"]
    predicted = decision["predicted_class"]
    stable = predicted if status == "matched" else None
    proposed = raw["candidate_class"] if stable is not None else candidate
    pending = proposed if stable is not None and proposed != stable else None
    pending_count = 1 if pending is not None else 0
    if status == "confirming":
        stable = None
        proposed = candidate
        pending = candidate
        pending_count = 1
    samples = 1 if raw["candidate_class"] is not None or stable is not None else 0
    decision["temporal"] = {
        "method": "vote",
        "window_size": 3,
        "samples": samples,
        "switch_confirmations": 2,
        "stable_class": stable,
        "proposed_class": proposed,
        "pending_class": pending,
        "pending_count": pending_count,
        "raw": {field: raw[field] for field in TEMPORAL_RAW_FIELDS},
    }


def shadow_record(
    source_id: str,
    major: str,
    yolo: str,
    stage2: str | None,
    decision: str | None,
    *,
    stage2_candidate: str | None = None,
    stage2_status: str | None = None,
    decision_status: str | None = None,
    decision_candidate: str | None = None,
    bbox: list[int] | None = None,
    input_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_candidate = stage2 if stage2_candidate is None else stage2_candidate
    raw_status = stage2_status or ("matched" if stage2 is not None else "unknown")
    final_status = decision_status or ("matched" if decision is not None else "unknown")
    final_candidate = decision_candidate
    if final_candidate is None and final_status not in {"no_face", "error"}:
        final_candidate = decision if decision is not None else raw_candidate
    raw_evidence = stage2_evidence(major, raw_candidate, stage2, raw_status)
    decision_evidence = stage2_evidence(
        major, final_candidate, decision, final_status
    )
    if any(
        raw_evidence[field] != decision_evidence[field]
        for field in STAGE2_CORE_FIELDS
    ):
        add_temporal_evidence(raw_evidence, decision_evidence)
    return {
        "schema": "recognition_shadow.v1",
        "timestamp": "2026-10-03T12:00:00.000+09:00",
        "input": dict(input_provenance or FILE_INPUT),
        "source": {
            "id": source_id,
            "frame_index": 1,
            "major_class": major,
            "confidence": 0.9,
            "bbox_xyxy": bbox or [0, 0, 10, 10],
            "raw_yolo_class_id": YOLO_CLASSES.index(yolo),
            "raw_yolo_class": yolo,
            "class_scores": yolo_scores(yolo),
        },
        "stage2_raw": raw_evidence,
        "decision": decision_evidence,
        "route": {"track_id": 1, "execution": "fresh"},
        "timing_ms": {"stage2": 3.5},
    }


def evaluate_records(
    records: dict[str, dict[str, object]],
    items: dict[str, dict[str, object]],
    *,
    shadow_input: dict[str, object] | None = None,
    golden_input: dict[str, object] | None = None,
) -> dict[str, object]:
    return evaluate(
        ShadowLog(
            input=dict(shadow_input or FILE_INPUT),
            records=records,  # type: ignore[arg-type]
        ),
        GoldenSet(
            input=dict(golden_input or FILE_INPUT),
            items=items,  # type: ignore[arg-type]
        ),
    )


class ShadowEvaluatorTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def write_golden(
        self,
        path: Path,
        items: list[dict[str, object]],
        *,
        input_provenance: dict[str, object] | None = None,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "recognition_golden.v1",
                    "input": dict(input_provenance or FILE_INPUT),
                    "items": items,
                }
            ),
            encoding="utf-8",
        )

    def fixture_files(self, directory: Path) -> tuple[Path, Path]:
        shadow_path = directory / "shadow.jsonl"
        golden_path = directory / "golden.json"
        records = [
            shadow_record("s1", "f", "f1", "f1", "f1"),
            shadow_record("s2", "f", "f3", "f1", "f2"),
            shadow_record(
                "s3",
                "k",
                "k1",
                None,
                None,
                stage2_candidate="k1",
                bbox=[5, 0, 15, 10],
            ),
            shadow_record("s4", "f", "f2", "f2", None),
            shadow_record("orphan", "k", "k2", "k2", "k2"),
        ]
        self.write_jsonl(shadow_path, records)
        self.write_golden(
            golden_path,
            [
                {
                    "id": "s1",
                    "expected_class": "f1",
                    "bbox_xyxy": [0, 0, 10, 10],
                },
                {"id": "s2", "expected_class": "f2"},
                {
                    "id": "s3",
                    "expected_class": "k1",
                    "bbox_xyxy": [0, 0, 10, 10],
                },
                {"id": "s4", "expected_class": "unknown"},
                {"id": "missing", "expected_class": "k2"},
            ],
        )
        return shadow_path, golden_path

    def test_evaluates_same_source_predictions_and_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path, golden_path = self.fixture_files(Path(value))
            result = evaluate(load_shadow(shadow_path), load_golden(golden_path))

        self.assertEqual(result["schema"], "recognition_shadow_evaluation.v1")
        self.assertEqual(result["input"], FILE_INPUT)
        self.assertEqual(
            result["counts"],
            {
                "shadow_records": 5,
                "golden_items": 5,
                "aligned": 4,
                "missing_shadow": 1,
                "missing_golden": 1,
                "invalid_records": 0,
            },
        )
        self.assertEqual(result["missing"]["shadow_for_golden"], ["missing"])
        self.assertEqual(result["missing"]["golden_for_shadow"], ["orphan"])
        self.assertEqual(result["coverage"]["alignment_rate"], 0.8)
        self.assertEqual(result["coverage"]["matched_rate"], 0.5)
        self.assertEqual(result["matched_accuracy"]["accuracy"], 1.0)
        self.assertEqual(result["accuracy"]["yolo"]["accuracy"], 0.666667)
        self.assertEqual(result["accuracy"]["stage2"]["accuracy"], 0.333333)
        self.assertEqual(
            result["accuracy"]["stage2_candidate"]["accuracy"], 0.666667
        )
        self.assertEqual(result["accuracy"]["decision"]["accuracy"], 0.666667)
        self.assertEqual(result["unknown_truth"]["decision_rejection_rate"], 1.0)

    def test_null_is_unlabeled_while_unknown_is_explicit_truth(self) -> None:
        records = {
            "unlabeled": shadow_record("unlabeled", "f", "f1", "f1", "f1"),
            "unknown": shadow_record("unknown", "f", "f2", None, None),
        }
        items = {
            "unlabeled": {
                "id": "unlabeled",
                "expected_class": None,
                "bbox_xyxy": [0, 0, 10, 10],
            },
            "unknown": {"id": "unknown", "expected_class": "unknown"},
        }
        result = evaluate_records(records, items)

        self.assertEqual(
            result["truth"],
            {"labeled": 1, "known": 0, "unknown": 1, "unlabeled": 1},
        )
        self.assertEqual(result["accuracy"]["decision"]["known_truth"], 0)
        self.assertIsNone(result["accuracy"]["decision"]["accuracy"])
        self.assertEqual(result["unknown_truth"]["items"], 1)
        self.assertEqual(result["unknown_truth"]["decision_rejected"], 1)
        self.assertEqual(result["bbox"]["compared"], 0)
        self.assertEqual(
            result["confusion"]["decision"]["per_class"],
            [{"expected": "unknown", "predicted": "unknown", "count": 1}],
        )

    def test_reports_agreement_confusion_bbox_and_major_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path, golden_path = self.fixture_files(Path(value))
            result = evaluate(load_shadow(shadow_path), load_golden(golden_path))

        self.assertEqual(result["agreement"]["yolo_stage2"]["exact_rate"], 0.5)
        self.assertEqual(
            result["agreement"]["stage2_decision"]["same_major_rate"], 0.75
        )
        self.assertEqual(result["bbox"]["compared"], 2)
        self.assertEqual(result["bbox"]["mean_iou"], 0.666667)
        self.assertEqual(result["bbox"]["iou_at_0_5"], 1)
        self.assertEqual(result["by_major"]["f"]["items"], 3)
        self.assertEqual(result["by_major"]["k"]["known_truth"], 1)
        decision_confusion = result["confusion"]["decision"]["per_class"]
        self.assertIn(
            {"expected": "k1", "predicted": "unknown", "count": 1},
            decision_confusion,
        )
        self.assertIn(
            {"expected": "unknown", "predicted": "unknown", "count": 1},
            decision_confusion,
        )

    def test_operational_error_and_wrong_major_route_remain_scored(self) -> None:
        error = shadow_record(
            "error",
            "f",
            "f1",
            None,
            None,
            stage2_status="error",
            decision_status="error",
        )
        truth_mismatch = shadow_record("truth", "f", "f1", "f1", "f1")
        records = {"error": error, "truth": truth_mismatch}
        items = {
            "error": {"id": "error", "expected_class": "f1"},
            "truth": {"id": "truth", "expected_class": "k1"},
        }
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"
            self.write_jsonl(shadow_path, list(records.values()))
            shadow = load_shadow(shadow_path)
        result = evaluate(
            shadow,
            GoldenSet(input=dict(FILE_INPUT), items=items),  # type: ignore[arg-type]
        )

        self.assertEqual(result["counts"]["invalid_records"], 2)
        self.assertEqual(result["invalid"]["major_mismatch"]["yolo"], 0)
        self.assertEqual(result["invalid"]["status_error"]["records"], 1)
        self.assertEqual(result["invalid"]["status_prediction_mismatch"]["ids"], [])
        self.assertEqual(
            result["invalid"]["truth_source_major_mismatch"]["ids"], ["truth"]
        )
        self.assertEqual(result["coverage"]["matched"], 1)
        self.assertEqual(result["accuracy"]["decision"]["known_truth"], 2)
        self.assertEqual(result["accuracy"]["decision"]["accuracy"], 0.0)

    def test_strictly_validates_fixed_score_domains_and_raw_class_id(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"
            wrong_yolo_domain = shadow_record("s1", "f", "f1", "f1", "f1")
            scores = wrong_yolo_domain["source"]["class_scores"]  # type: ignore[index]
            del scores["f10"]  # type: ignore[index]
            scores["fake"] = 0.1  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_yolo_domain])
            with self.assertRaisesRegex(EvaluationInputError, "missing f10.*unexpected fake"):
                load_shadow(shadow_path)

            wrong_stage2_domain = shadow_record("s1", "f", "f1", "f1", "f1")
            scores = wrong_stage2_domain["stage2_raw"]["class_scores"]  # type: ignore[index]
            del scores["f10"]  # type: ignore[index]
            scores["k1"] = 0.1  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_stage2_domain])
            with self.assertRaisesRegex(EvaluationInputError, "missing f10.*unexpected k1"):
                load_shadow(shadow_path)

            wrong_id = shadow_record("s1", "f", "f1", "f1", "f1")
            wrong_id["source"]["raw_yolo_class_id"] = 20  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_id])
            with self.assertRaisesRegex(EvaluationInputError, "does not match"):
                load_shadow(shadow_path)

    def test_rejects_impossible_status_and_prediction_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"
            cases: list[tuple[str, dict[str, object], str]] = []

            raw_passthrough = shadow_record("s1", "f", "f1", "f1", "f1")
            raw_passthrough["stage2_raw"]["status"] = "passthrough"  # type: ignore[index]
            cases.append(("raw passthrough", raw_passthrough, "status must be one of"))

            final_unknown_status = shadow_record("s1", "f", "f1", "f1", "f1")
            final_unknown_status["decision"]["status"] = "other"  # type: ignore[index]
            cases.append(
                ("unknown final status", final_unknown_status, "status must be one of")
            )

            knife_no_face = shadow_record(
                "s1",
                "k",
                "k1",
                None,
                None,
                stage2_status="error",
                decision_status="error",
            )
            knife_no_face["stage2_raw"]["status"] = "no_face"  # type: ignore[index]
            cases.append(("knife no_face", knife_no_face, "status must be one of"))

            predicted_while_confirming = shadow_record(
                "s1", "f", "f1", "f1", "f1"
            )
            predicted_while_confirming["decision"]["status"] = "confirming"  # type: ignore[index]
            cases.append(
                (
                    "prediction while confirming",
                    predicted_while_confirming,
                    "requires predicted_class to be null",
                )
            )

            unknown_without_scores = shadow_record(
                "s1",
                "f",
                "f1",
                None,
                None,
                stage2_status="error",
                decision_status="error",
            )
            unknown_without_scores["stage2_raw"]["status"] = "unknown"  # type: ignore[index]
            cases.append(
                ("unknown without scores", unknown_without_scores, "requires a populated")
            )

            for name, record, message in cases:
                with self.subTest(name=name):
                    self.write_jsonl(shadow_path, [record])
                    with self.assertRaisesRegex(EvaluationInputError, message):
                        load_shadow(shadow_path)

    def test_rejects_inconsistent_yolo_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"

            wrong_major = shadow_record("s1", "f", "k1", "f1", "f1")
            self.write_jsonl(shadow_path, [wrong_major])
            with self.assertRaisesRegex(EvaluationInputError, "major_class does not match"):
                load_shadow(shadow_path)

            wrong_argmax = shadow_record("s1", "f", "f1", "f1", "f1")
            wrong_argmax["source"]["class_scores"]["f2"] = 0.95  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_argmax])
            with self.assertRaisesRegex(EvaluationInputError, "not a maximum-scoring"):
                load_shadow(shadow_path)

            wrong_confidence = shadow_record("s1", "f", "f1", "f1", "f1")
            wrong_confidence["source"]["confidence"] = 0.8  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_confidence])
            with self.assertRaisesRegex(EvaluationInputError, "confidence does not match"):
                load_shadow(shadow_path)

    def test_rejects_inconsistent_stage2_score_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"

            raw_not_argmax = shadow_record("s1", "f", "f1", "f1", "f1")
            raw_not_argmax["stage2_raw"]["class_scores"]["f2"] = 0.95  # type: ignore[index]
            self.write_jsonl(shadow_path, [raw_not_argmax])
            with self.assertRaisesRegex(EvaluationInputError, "candidate_class is not"):
                load_shadow(shadow_path)

            wrong_score = shadow_record("s1", "f", "f1", "f1", "f1")
            wrong_score["stage2_raw"]["score"] = 0.8  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_score])
            with self.assertRaisesRegex(EvaluationInputError, "score does not match"):
                load_shadow(shadow_path)

            wrong_second = shadow_record("s1", "f", "f1", "f1", "f1")
            raw = wrong_second["stage2_raw"]  # type: ignore[assignment]
            raw["second_best_class"] = "f2"  # type: ignore[index]
            raw["second_best_score"] = raw["class_scores"]["f2"]  # type: ignore[index]
            raw["margin"] = raw["score"] - raw["second_best_score"]  # type: ignore[index,operator]
            self.write_jsonl(shadow_path, [wrong_second])
            with self.assertRaisesRegex(EvaluationInputError, "not the highest-scoring"):
                load_shadow(shadow_path)

            wrong_margin = shadow_record("s1", "f", "f1", "f1", "f1")
            wrong_margin["stage2_raw"]["margin"] = 0.5  # type: ignore[index]
            self.write_jsonl(shadow_path, [wrong_margin])
            with self.assertRaisesRegex(EvaluationInputError, "margin does not equal"):
                load_shadow(shadow_path)

    def test_final_temporal_candidate_may_be_non_argmax_but_stays_consistent(self) -> None:
        record = shadow_record("s1", "f", "f1", "f2", "f1")
        decision = record["decision"]  # type: ignore[assignment]
        scores = decision["class_scores"]  # type: ignore[index]
        scores["f10"] = 0.95  # type: ignore[index]
        decision["second_best_class"] = "f10"  # type: ignore[index]
        decision["second_best_score"] = 0.95  # type: ignore[index]
        decision["margin"] = decision["score"] - 0.95  # type: ignore[index,operator]

        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"
            self.write_jsonl(shadow_path, [record])
            loaded = load_shadow(shadow_path)

            inconsistent = json.loads(json.dumps(record))
            inconsistent["decision"]["score"] = 0.8
            self.write_jsonl(shadow_path, [inconsistent])
            with self.assertRaisesRegex(EvaluationInputError, "score does not match"):
                load_shadow(shadow_path)

        self.assertEqual(loaded.records["s1"]["decision"]["candidate_class"], "f1")

    def test_validates_route_and_unsmoothed_execution_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"

            direct = shadow_record("s1", "f", "f1", "f1", "f1")
            direct["route"] = {"track_id": None, "execution": "direct"}
            self.write_jsonl(shadow_path, [direct])
            self.assertEqual(load_shadow(shadow_path).records["s1"]["route"], direct["route"])

            direct_with_track = json.loads(json.dumps(direct))
            direct_with_track["route"]["track_id"] = 1
            self.write_jsonl(shadow_path, [direct_with_track])
            with self.assertRaisesRegex(EvaluationInputError, "requires track_id to be null"):
                load_shadow(shadow_path)

            direct_with_temporal = json.loads(json.dumps(direct))
            add_temporal_evidence(
                direct_with_temporal["stage2_raw"],  # type: ignore[arg-type]
                direct_with_temporal["decision"],  # type: ignore[arg-type]
            )
            self.write_jsonl(shadow_path, [direct_with_temporal])
            with self.assertRaisesRegex(EvaluationInputError, "must not contain temporal"):
                load_shadow(shadow_path)

            direct_changed = shadow_record("s1", "f", "f1", "f1", "f2")
            direct_changed["route"] = {"track_id": None, "execution": "direct"}
            del direct_changed["decision"]["temporal"]  # type: ignore[index]
            self.write_jsonl(shadow_path, [direct_changed])
            with self.assertRaisesRegex(EvaluationInputError, "differs from stage2_raw"):
                load_shadow(shadow_path)

            for track_id in (None, 0):
                with self.subTest(fresh_track_id=track_id):
                    fresh = shadow_record("s1", "f", "f1", "f1", "f1")
                    fresh["route"]["track_id"] = track_id  # type: ignore[index]
                    self.write_jsonl(shadow_path, [fresh])
                    with self.assertRaisesRegex(
                        EvaluationInputError, "requires a positive track_id"
                    ):
                        load_shadow(shadow_path)

            unsmoothed_changed = shadow_record("s1", "f", "f1", "f1", "f2")
            del unsmoothed_changed["decision"]["temporal"]  # type: ignore[index]
            self.write_jsonl(shadow_path, [unsmoothed_changed])
            with self.assertRaisesRegex(EvaluationInputError, "without temporal smoothing"):
                load_shadow(shadow_path)

    def test_validates_temporal_raw_provenance_and_state(self) -> None:
        record = shadow_record("s1", "f", "f1", "f2", "f1")
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"
            self.write_jsonl(shadow_path, [record])
            load_shadow(shadow_path)

            held_after_no_face = shadow_record(
                "held",
                "f",
                "f1",
                None,
                "f1",
                stage2_status="no_face",
            )
            self.write_jsonl(shadow_path, [held_after_no_face])
            load_shadow(shadow_path)

            smoothed_no_face = shadow_record(
                "rejected",
                "f",
                "f1",
                None,
                None,
                stage2_status="no_face",
                decision_status="no_face",
            )
            add_temporal_evidence(
                smoothed_no_face["stage2_raw"],  # type: ignore[arg-type]
                smoothed_no_face["decision"],  # type: ignore[arg-type]
            )
            self.write_jsonl(shadow_path, [smoothed_no_face])
            load_shadow(shadow_path)

            cases: list[tuple[str, dict[str, object], str]] = []

            raw_mismatch = json.loads(json.dumps(record))
            raw_mismatch["decision"]["temporal"]["raw"]["candidate_class"] = "f3"
            cases.append(("raw mismatch", raw_mismatch, "differs from stage2_raw"))

            invalid_method = json.loads(json.dumps(record))
            invalid_method["decision"]["temporal"]["method"] = "median"
            cases.append(("invalid method", invalid_method, "method must be mean or vote"))

            invalid_samples = json.loads(json.dumps(record))
            invalid_samples["decision"]["temporal"]["samples"] = 4
            cases.append(("invalid samples", invalid_samples, "between zero and window_size"))

            invalid_pending = json.loads(json.dumps(record))
            invalid_pending["decision"]["temporal"]["pending_count"] = 0
            cases.append(("invalid pending", invalid_pending, "pending_class and pending_count"))

            invalid_stable = json.loads(json.dumps(record))
            invalid_stable["decision"]["temporal"]["stable_class"] = "f3"
            cases.append(("invalid stable", invalid_stable, "match the accepted decision"))

            invalid_score_kind = json.loads(json.dumps(record))
            invalid_score_kind["decision"]["score_kind"] = "arbitrary_score"
            cases.append(
                (
                    "invalid score kind",
                    invalid_score_kind,
                    "outside the fixed f score kind set",
                )
            )

            for name, malformed, message in cases:
                with self.subTest(name=name):
                    self.write_jsonl(shadow_path, [malformed])
                    with self.assertRaisesRegex(EvaluationInputError, message):
                        load_shadow(shadow_path)

            knife_score_mismatch = shadow_record("s1", "k", "k1", "k1", "k2")
            knife_score_mismatch["decision"]["score_kind"] = (  # type: ignore[index]
                "cosine_similarity"
            )
            self.write_jsonl(shadow_path, [knife_score_mismatch])
            with self.assertRaisesRegex(EvaluationInputError, "must match stage2_raw"):
                load_shadow(shadow_path)

            empty_status_changed = shadow_record(
                "s1",
                "f",
                "f1",
                None,
                None,
                stage2_status="no_face",
                decision_status="error",
            )
            self.write_jsonl(shadow_path, [empty_status_changed])
            with self.assertRaisesRegex(EvaluationInputError, "differs from stage2_raw"):
                load_shadow(shadow_path)

            confirming_without_history = shadow_record(
                "s1",
                "f",
                "f1",
                "f1",
                None,
                decision_status="confirming",
                decision_candidate="f1",
            )
            confirming_without_history["decision"]["temporal"][  # type: ignore[index]
                "samples"
            ] = 0
            self.write_jsonl(shadow_path, [confirming_without_history])
            with self.assertRaisesRegex(EvaluationInputError, "samples must be positive"):
                load_shadow(shadow_path)

    def test_rejects_duplicate_ids_wrong_schema_and_illegal_golden_label(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            shadow_path = directory / "shadow.jsonl"
            duplicate = shadow_record("same", "f", "f1", "f1", "f1")
            self.write_jsonl(shadow_path, [duplicate, duplicate])
            with self.assertRaisesRegex(EvaluationInputError, "duplicate shadow"):
                load_shadow(shadow_path)

            invalid_schema = shadow_record("s1", "f", "f1", "f1", "f1")
            invalid_schema["schema"] = "recognition_shadow.v0"
            self.write_jsonl(shadow_path, [invalid_schema])
            with self.assertRaisesRegex(
                EvaluationInputError, "expected 'recognition_shadow.v1'"
            ):
                load_shadow(shadow_path)

            golden_path = directory / "golden.json"
            self.write_golden(
                golden_path,
                [
                    {"id": "same", "expected_class": "f1"},
                    {"id": "same", "expected_class": "f2"},
                ],
            )
            with self.assertRaisesRegex(EvaluationInputError, "duplicate golden"):
                load_golden(golden_path)

            self.write_golden(
                golden_path, [{"id": "s1", "expected_class": "person"}]
            )
            with self.assertRaisesRegex(EvaluationInputError, "fixed golden class"):
                load_golden(golden_path)

    def test_validates_and_matches_file_and_capture_input_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            shadow_path = directory / "shadow.jsonl"
            golden_path = directory / "golden.json"
            capture_record = shadow_record(
                "s1",
                "f",
                "f1",
                "f1",
                "f1",
                input_provenance=CAPTURE_INPUT,
            )
            self.write_jsonl(shadow_path, [capture_record])
            self.write_golden(
                golden_path,
                [{"id": "s1", "expected_class": "f1"}],
                input_provenance=CAPTURE_INPUT,
            )
            result = evaluate(load_shadow(shadow_path), load_golden(golden_path))
            self.assertEqual(result["input"], CAPTURE_INPUT)

            inconsistent = shadow_record("s2", "f", "f2", "f2", "f2")
            inconsistent["input"]["sha256"] = "b" * 64  # type: ignore[index]
            self.write_jsonl(
                shadow_path,
                [shadow_record("s1", "f", "f1", "f1", "f1"), inconsistent],
            )
            with self.assertRaisesRegex(EvaluationInputError, "differs from earlier"):
                load_shadow(shadow_path)

            extra_field = shadow_record("s1", "f", "f1", "f1", "f1")
            extra_field["input"]["device"] = "D455"  # type: ignore[index]
            self.write_jsonl(shadow_path, [extra_field])
            with self.assertRaisesRegex(EvaluationInputError, "unexpected field.*device"):
                load_shadow(shadow_path)

        with self.assertRaisesRegex(EvaluationInputError, "provenance does not match"):
            evaluate_records(
                {"s1": shadow_record("s1", "f", "f1", "f1", "f1")},
                {"s1": {"id": "s1", "expected_class": "f1"}},
                golden_input={**FILE_INPUT, "video_stride": 2},
            )

    def test_rejects_malformed_input_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            shadow_path = Path(value) / "shadow.jsonl"
            malformed = shadow_record("s1", "f", "f1", "f1", "f1")
            malformed["input"]["sha256"] = "A" * 64  # type: ignore[index]
            self.write_jsonl(shadow_path, [malformed])
            with self.assertRaisesRegex(EvaluationInputError, "lowercase hexadecimal"):
                load_shadow(shadow_path)

            malformed = shadow_record("s1", "f", "f1", "f1", "f1")
            malformed["input"] = {
                "kind": "capture",
                "capture_id": "",
                "video_stride": 1,
            }
            self.write_jsonl(shadow_path, [malformed])
            with self.assertRaisesRegex(EvaluationInputError, "non-empty string"):
                load_shadow(shadow_path)

    def test_cli_writes_machine_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            shadow_path, golden_path = self.fixture_files(directory)
            output_path = directory / "nested" / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "tools" / "evaluation" / "evaluate_shadow.py"),
                    "--shadow",
                    str(shadow_path),
                    "--golden",
                    str(golden_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("aligned: 4/5", completed.stdout)
        self.assertEqual(report["counts"]["aligned"], 4)


if __name__ == "__main__":
    unittest.main()
