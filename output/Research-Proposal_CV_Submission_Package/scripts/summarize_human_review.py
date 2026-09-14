#!/usr/bin/env python3
"""Validate two blinded review files, calculate agreement, and adjudicate disagreements.

Run from the repository root:
    python scripts/summarize_human_review.py

After completing the final_* columns in human_review_adjudication.csv:
    python scripts/summarize_human_review.py --finalize
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


RATING_COLUMNS = ("object_correct", "non_informative", "hallucination")
IDENTITY_COLUMNS = ("review_id", "image_file", "caption")
FINAL_COLUMNS = tuple(f"final_{name}" for name in RATING_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("results/human_review"),
        help="Directory containing the completed review CSV files.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=336,
        help="Expected number of rows in each reviewer packet.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Create final consensus results after adjudication columns are filled.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"ERROR: missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"ERROR: no header found in {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_packet(path: Path, fields: list[str], rows: list[dict[str, str]], expected: int) -> None:
    required = set(IDENTITY_COLUMNS + RATING_COLUMNS)
    missing = sorted(required.difference(fields))
    if missing:
        raise SystemExit(f"ERROR: {path.name} is missing columns: {', '.join(missing)}")
    if len(rows) != expected:
        raise SystemExit(f"ERROR: {path.name} contains {len(rows)} rows; expected {expected}")
    ids = [row["review_id"].strip() for row in rows]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"ERROR: {path.name} contains duplicate review_id values")
    for line_number, row in enumerate(rows, start=2):
        for column in RATING_COLUMNS:
            if row[column].strip() not in {"0", "1"}:
                raise SystemExit(
                    f"ERROR: {path.name}, row {line_number}, {column} must contain 0 or 1"
                )


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float, float]:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    a_positive = sum(a) / n
    b_positive = sum(b) / n
    expected = a_positive * b_positive + (1 - a_positive) * (1 - b_positive)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else math.nan
    return observed, expected, kappa


def add_key(review_dir: Path, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    key_path = review_dir / "unblinding_key.csv"
    if not key_path.exists():
        return rows, []
    key_fields, key_rows = read_csv(key_path)
    if "review_id" not in key_fields:
        raise SystemExit("ERROR: unblinding_key.csv has no review_id column")
    key_by_id = {row["review_id"].strip(): row for row in key_rows}
    if len(key_by_id) != len(key_rows):
        raise SystemExit("ERROR: unblinding_key.csv contains duplicate review_id values")
    missing = [str(row["review_id"]) for row in rows if str(row["review_id"]) not in key_by_id]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} review IDs are absent from unblinding_key.csv")
    metadata = [name for name in key_fields if name != "review_id"]
    for row in rows:
        row.update({name: key_by_id[str(row["review_id"])].get(name, "") for name in metadata})
    return rows, metadata


def build_outputs(
    review_dir: Path, expected: int, write_adjudication: bool = True
) -> tuple[list[dict[str, object]], list[str]]:
    path_a = review_dir / "rater_a_completed.csv"
    path_b = review_dir / "rater_b_completed.csv"
    fields_a, rows_a = read_csv(path_a)
    fields_b, rows_b = read_csv(path_b)
    validate_packet(path_a, fields_a, rows_a, expected)
    validate_packet(path_b, fields_b, rows_b, expected)

    by_id_b = {row["review_id"].strip(): row for row in rows_b}
    ids_a = {row["review_id"].strip() for row in rows_a}
    ids_b = set(by_id_b)
    if ids_a != ids_b:
        raise SystemExit(
            f"ERROR: reviewer ID sets differ (only A: {len(ids_a - ids_b)}, only B: {len(ids_b - ids_a)})"
        )

    merged: list[dict[str, object]] = []
    for row_a in rows_a:
        review_id = row_a["review_id"].strip()
        row_b = by_id_b[review_id]
        for column in ("image_file", "caption"):
            if row_a[column] != row_b[column]:
                raise SystemExit(f"ERROR: {column} differs between reviewers for {review_id}")
        row: dict[str, object] = {column: row_a[column] for column in IDENTITY_COLUMNS}
        for column in RATING_COLUMNS:
            value_a = int(row_a[column])
            value_b = int(row_b[column])
            row[f"{column}_a"] = value_a
            row[f"{column}_b"] = value_b
            row[f"{column}_agree"] = int(value_a == value_b)
            row[f"final_{column}"] = str(value_a) if value_a == value_b else ""
        row["any_disagreement"] = int(
            any(row[f"{column}_a"] != row[f"{column}_b"] for column in RATING_COLUMNS)
        )
        row["adjudication_notes"] = ""
        merged.append(row)

    # The key is opened only after the completed packets have passed validation.
    merged, metadata = add_key(review_dir, merged)

    summary_rows: list[dict[str, object]] = []
    for column in RATING_COLUMNS:
        a = [int(row[f"{column}_a"]) for row in merged]
        b = [int(row[f"{column}_b"]) for row in merged]
        observed, expected_agreement, kappa = cohen_kappa(a, b)
        summary_rows.append(
            {
                "metric": column,
                "n": len(a),
                "rater_a_positive": sum(a),
                "rater_b_positive": sum(b),
                "agreements": sum(x == y for x, y in zip(a, b)),
                "disagreements": sum(x != y for x, y in zip(a, b)),
                "raw_agreement": f"{observed:.6f}",
                "expected_agreement": f"{expected_agreement:.6f}",
                "cohen_kappa": f"{kappa:.6f}",
            }
        )

    summary_fields = [
        "metric", "n", "rater_a_positive", "rater_b_positive", "agreements",
        "disagreements", "raw_agreement", "expected_agreement", "cohen_kappa",
    ]
    write_csv(review_dir / "agreement_summary.csv", summary_fields, summary_rows)

    rating_fields: list[str] = []
    for column in RATING_COLUMNS:
        rating_fields.extend([f"{column}_a", f"{column}_b", f"{column}_agree"])
    output_fields = [
        "review_id", *metadata, "image_file", "caption", *rating_fields,
        "any_disagreement", *FINAL_COLUMNS, "adjudication_notes",
    ]
    disagreements = [row for row in merged if row["any_disagreement"] == 1]
    if write_adjudication:
        write_csv(review_dir / "human_review_adjudication.csv", output_fields, disagreements)
    write_csv(review_dir / "human_review_consensus_draft.csv", output_fields, merged)

    print(f"PASS: reviewer A has {len(rows_a)} complete unique rows")
    print(f"PASS: reviewer B has {len(rows_b)} complete unique rows")
    print(f"PASS: reviewer ID sets and caption metadata match")
    print("\nAgreement:")
    for row in summary_rows:
        print(
            f"  {row['metric']}: {100 * float(row['raw_agreement']):.1f}% raw agreement, "
            f"kappa={float(row['cohen_kappa']):.3f}, disagreements={row['disagreements']}"
        )
    print(f"\nRows requiring adjudication: {len(disagreements)}")
    print(f"Wrote: {review_dir / 'agreement_summary.csv'}")
    if write_adjudication:
        print(f"Wrote: {review_dir / 'human_review_adjudication.csv'}")
    else:
        print(f"Preserved: {review_dir / 'human_review_adjudication.csv'}")
    print(f"Wrote: {review_dir / 'human_review_consensus_draft.csv'}")
    return merged, output_fields


def finalize(review_dir: Path, merged: list[dict[str, object]], output_fields: list[str]) -> None:
    adjudication_path = review_dir / "human_review_adjudication.csv"
    adjudication_fields, adjudication_rows = read_csv(adjudication_path)
    missing_columns = sorted(set(FINAL_COLUMNS).difference(adjudication_fields))
    if missing_columns:
        raise SystemExit(f"ERROR: adjudication file is missing: {', '.join(missing_columns)}")

    expected_ids = {str(row["review_id"]) for row in merged if row["any_disagreement"] == 1}
    actual_ids = {row["review_id"].strip() for row in adjudication_rows}
    if actual_ids != expected_ids:
        raise SystemExit("ERROR: adjudication rows do not match the current disagreement set")
    by_id = {row["review_id"].strip(): row for row in adjudication_rows}
    for line_number, row in enumerate(adjudication_rows, start=2):
        for column in FINAL_COLUMNS:
            if row[column].strip() not in {"0", "1"}:
                raise SystemExit(
                    f"ERROR: {adjudication_path.name}, row {line_number}, {column} must contain 0 or 1"
                )

    final_rows: list[dict[str, object]] = []
    for row in merged:
        final_row = dict(row)
        review_id = str(row["review_id"])
        if review_id in by_id:
            source = by_id[review_id]
            for column in FINAL_COLUMNS:
                final_row[column] = source[column].strip()
            final_row["adjudication_notes"] = source.get("adjudication_notes", "")
        final_rows.append(final_row)

    write_csv(review_dir / "human_review_consensus_final.csv", output_fields, final_rows)

    groups: list[tuple[str, list[dict[str, object]]]] = [("overall", final_rows)]
    for field in ("condition", "income_quartile"):
        if field in output_fields:
            for value in sorted({str(row.get(field, "")) for row in final_rows}):
                groups.append((f"{field}={value}", [row for row in final_rows if str(row.get(field, "")) == value]))
    if "condition" in output_fields and "income_quartile" in output_fields:
        combinations = sorted({(str(row.get("condition", "")), str(row.get("income_quartile", ""))) for row in final_rows})
        for condition, quartile in combinations:
            groups.append(
                (
                    f"condition={condition};income_quartile={quartile}",
                    [row for row in final_rows if str(row.get("condition", "")) == condition and str(row.get("income_quartile", "")) == quartile],
                )
            )

    metric_rows: list[dict[str, object]] = []
    for group_name, rows in groups:
        item: dict[str, object] = {"group": group_name, "n": len(rows)}
        for column in RATING_COLUMNS:
            values = [int(str(row[f"final_{column}"])) for row in rows]
            item[f"{column}_count"] = sum(values)
            item[f"{column}_rate"] = f"{sum(values) / len(values):.6f}" if values else ""
        metric_rows.append(item)
    metric_fields = ["group", "n"]
    for column in RATING_COLUMNS:
        metric_fields.extend([f"{column}_count", f"{column}_rate"])
    write_csv(review_dir / "human_review_final_metrics.csv", metric_fields, metric_rows)

    print("\nFINALIZED:")
    print(f"Wrote: {review_dir / 'human_review_consensus_final.csv'}")
    print(f"Wrote: {review_dir / 'human_review_final_metrics.csv'}")


def main() -> None:
    args = parse_args()
    merged, fields = build_outputs(
        args.review_dir, args.expected_rows, write_adjudication=not args.finalize
    )
    if args.finalize:
        finalize(args.review_dir, merged, fields)


if __name__ == "__main__":
    main()
