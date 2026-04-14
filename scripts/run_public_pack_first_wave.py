#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.block3.models.registry import check_model_available  # noqa: E402
from narrative.public_pack import (  # noqa: E402
    FIRST_WAVE_MODEL_PRESETS,
    SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES,
    filter_public_pack_cells,
    load_public_pack_registry,
    prepare_public_pack_data,
    resolve_public_pack_downloads,
    resolve_public_pack_models,
    run_public_pack_models,
    stage_public_pack_downloads,
)
from narrative.public_pack.registry import expand_public_pack_cells  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run public-pack first-wave entrants with availability-aware staging."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pack", default="")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prediction-length", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--preset", default="first_wave_entrants")
    parser.add_argument("--stage-supported-datasets", action="store_true")
    parser.add_argument("--overwrite-datasets", action="store_true")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-entities", type=int, default=None)
    parser.add_argument("--max-train-samples-per-entity", type=int, default=512)
    parser.add_argument("--max-covariates", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fail-on-unavailable-models", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--list-supported-datasets", action="store_true")
    return parser.parse_args()


def _split_supported_stage_families(selected_families: Sequence[str]) -> Tuple[List[str], List[str]]:
    supported = sorted({family for family in selected_families if family in SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES})
    unsupported = sorted({family for family in selected_families if family not in SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES})
    return supported, unsupported


def _partition_available_models(
    model_names: Sequence[str],
    availability_fn: Callable[[str], bool] = check_model_available,
) -> Tuple[List[str], List[str]]:
    available: List[str] = []
    unavailable: List[str] = []
    for model_name in model_names:
        if availability_fn(model_name):
            available.append(model_name)
        else:
            unavailable.append(model_name)
    return available, unavailable


def _cell_slug(family: str, variant: str, context_length: int, prediction_length: int) -> str:
    base = f"{family}__{variant}__ctx{context_length}__pred{prediction_length}"
    return base.replace("/", "_")


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    if args.max_entities is None:
        args.max_entities = 3
    if args.max_train_samples_per_entity == 512:
        args.max_train_samples_per_entity = 64
    if args.max_models is None:
        args.max_models = 2


def main() -> int:
    args = _parse_args()

    if args.list_presets:
        print(json.dumps(FIRST_WAVE_MODEL_PRESETS, indent=2))
        return 0
    if args.list_supported_datasets:
        print(json.dumps({"supported_families": list(SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES)}, indent=2))
        return 0

    _apply_smoke_defaults(args)
    registry = load_public_pack_registry(args.config)
    cells = expand_public_pack_cells(
        registry,
        pack=args.pack,
        requested_families=args.family,
    )
    cells = filter_public_pack_cells(
        cells,
        requested_variants=args.variant,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    if not cells:
        raise ValueError("no public-pack cells matched the requested filters")

    selected_families = sorted({cell.family for cell in cells})
    supported_stage_families, unsupported_stage_families = _split_supported_stage_families(selected_families)
    stage_rows = []
    if args.stage_supported_datasets and supported_stage_families:
        downloads = resolve_public_pack_downloads(registry, requested_families=supported_stage_families)
        stage_rows = stage_public_pack_downloads(downloads, overwrite=args.overwrite_datasets)

    all_results = []
    prediction_frames = []
    cell_summaries = []

    for cell in cells:
        prepared = prepare_public_pack_data(
            cell,
            dataset_path=args.dataset_path,
            target=args.target,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            max_entities=args.max_entities,
            max_train_samples_per_entity=args.max_train_samples_per_entity,
            max_covariates=args.max_covariates,
        )
        requested_models = resolve_public_pack_models(
            args.model,
            preset=args.preset,
            is_binary_target=prepared.is_binary_target,
        )
        available_models, unavailable_models = _partition_available_models(requested_models)
        if args.max_models is not None:
            available_models = available_models[: args.max_models]
        if unavailable_models and args.fail_on_unavailable_models:
            raise ValueError(
                f"unavailable first-wave models for family={cell.family}, variant={cell.variant}: {unavailable_models}"
            )
        if not available_models:
            raise ValueError(
                f"no available first-wave models for family={cell.family}, variant={cell.variant}; requested={requested_models}"
            )

        results, predictions = run_public_pack_models(
            prepared,
            available_models,
            n_bootstrap=args.n_bootstrap,
        )
        all_results.extend(results)
        if not predictions.empty:
            prediction_frames.append(predictions)
        cell_summaries.append(
            {
                "pack": cell.pack,
                "family": cell.family,
                "variant": cell.variant,
                "dataset_path": str(prepared.dataset_path),
                "context_length": prepared.context_length,
                "prediction_length": prepared.prediction_length,
                "n_entities": prepared.metadata.get("n_entities", 0),
                "n_train_samples": prepared.metadata.get("n_train_samples", 0),
                "n_test_samples": prepared.metadata.get("n_test_samples", 0),
                "requested_models": requested_models,
                "executed_models": available_models,
                "unavailable_models": unavailable_models,
            }
        )

        if args.output_dir is not None:
            cell_output_dir = args.output_dir / _cell_slug(
                cell.family,
                cell.variant,
                prepared.context_length,
                prepared.prediction_length,
            )
            cell_output_dir.mkdir(parents=True, exist_ok=True)
            (cell_output_dir / "metrics.json").write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            if not predictions.empty:
                predictions.to_parquet(cell_output_dir / "predictions.parquet", index=False)

    payload = {
        "selection": {
            "pack": args.pack or None,
            "families": args.family,
            "variants": args.variant,
            "preset": args.preset,
            "models": args.model,
            "smoke": bool(args.smoke),
        },
        "staging": {
            "stage_supported_datasets": bool(args.stage_supported_datasets),
            "supported_stage_families": supported_stage_families,
            "unsupported_stage_families": unsupported_stage_families,
            "rows": stage_rows,
        },
        "cells": cell_summaries,
        "results": all_results,
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "MANIFEST.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if prediction_frames:
            pd.concat(prediction_frames, ignore_index=True).to_parquet(
                args.output_dir / "predictions.parquet",
                index=False,
            )

    summary = {
        "n_cells": len(cell_summaries),
        "n_results": len(all_results),
        "n_errors": sum(1 for result in all_results if "error" in result),
        "families": sorted({result["family"] for result in all_results}) if all_results else [],
        "models": sorted({result["model_name"] for result in all_results}) if all_results else [],
        "staged_families": supported_stage_families if args.stage_supported_datasets else [],
        "unsupported_stage_families": unsupported_stage_families,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())