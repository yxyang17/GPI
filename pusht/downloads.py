"""Utilities to lazily download datasets/checkpoints from Google Drive."""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import gdown

MODELS_DIR = Path("models")

_RESOURCE_MAP: Dict[str, str] = {
    "pusht_cchi_v7_replay.zarr.zip": "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t",
    "GPI_vision_model.zip": "1eq2U6Ztt1uWfucFMnEcmGJAFRzWzxc5o",
}

_BUNDLE_CONTENTS: Dict[str, Tuple[str, ...]] = {
    "GPI_vision_model.zip": (
        "pusht_cchi_v7_replay_imgs_feature_epoch_200.zarr",
        "vision_state_predictor_epoch_200.ckpt",
    ),
}

_BUNDLE_LOOKUP: Dict[str, str] = {
    member: archive for archive, members in _BUNDLE_CONTENTS.items() for member in members
}


def _is_within_models(path: Path) -> bool:
    try:
        path.relative_to(MODELS_DIR)
        return True
    except ValueError:
        return False


def _resolve_target(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if _is_within_models(raw):
        return raw
    return MODELS_DIR / raw


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _download(path: Path, file_id: str) -> None:
    _ensure_models_dir()
    print(f"[download] fetching {path.name} -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(path), quiet=False)
    if not path.exists():
        raise RuntimeError(f"Download failed for '{path.name}'")
    print("[download] complete")


def _extract_bundle(archive_path: Path, expected_members: Iterable[str]) -> None:
    _ensure_models_dir()
    print(f"[extract] unpacking {archive_path.name} -> {archive_path.parent}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(archive_path.parent)
    for member in expected_members:
        target = archive_path.parent / member
        if target.exists():
            continue
        matches = list(archive_path.parent.rglob(member))
        for candidate in matches:
            if candidate == target:
                break
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(candidate), str(target))
            break
        if not target.exists():
            raise FileNotFoundError(
                f"Expected '{member}' after extracting {archive_path.name}"
            )


def ensure_resource(path: str) -> str:
    """Ensure `path` exists locally, downloading it from Drive if needed."""
    target_path = _resolve_target(path)
    legacy_path = Path(path)
    if legacy_path != target_path and legacy_path.exists():
        _ensure_models_dir()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(target_path))
        return str(target_path)
    if target_path.exists():
        return str(target_path)

    filename = target_path.name
    bundle_name = _BUNDLE_LOOKUP.get(filename)
    if bundle_name is not None:
        archive_path = target_path.parent / bundle_name
        ensure_resource(str(archive_path))
        _extract_bundle(archive_path, _BUNDLE_CONTENTS[bundle_name])
        if target_path.exists():
            return str(target_path)
        raise FileNotFoundError(
            f"Resource '{path}' not found after extracting bundle '{bundle_name}'."
        )

    file_id = _RESOURCE_MAP.get(filename)
    if file_id is None:
        raise FileNotFoundError(
            f"Resource '{path}' not found and no download id registered."
        )

    _download(target_path, file_id)
    return str(target_path)


__all__ = ["ensure_resource"]
