from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import torch


MAIN_SHEET = "main"
DEFAULT_MAX_PRIMITIVES = 17
DEFAULT_TEXT_MODEL = "google/siglip-base-patch16-224"

SPREADSHEET_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CELL_RE = re.compile(r"([A-Z]+)")


@dataclass(frozen=True)
class PromptRecord:
    task_id: str
    task_meta_text: str
    primitive_chain: tuple[str, ...]
    prompt: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def format_prompt(task_meta_text: str, primitive_chain: Iterable[str]) -> str:
    chain = [clean_text(item) for item in primitive_chain if clean_text(item)]
    chain_text = " -> ".join(chain) if chain else "No canonical primitive chain provided."
    return "\n".join(
        [
            "You are evaluating a short robot manipulation segment.",
            "",
            "High-level task goal:",
            clean_text(task_meta_text),
            "",
            "Canonical primitive chain for this task:",
            chain_text,
            "",
            "Question:",
            "Given the current observation history, robot proprioception, and candidate future action chunk, "
            "estimate the progress increment of the current local primitive only. Do not estimate progress "
            "toward the whole task. The current primitive label is not provided; infer the active primitive "
            "from the observation and action.",
            "",
            "Output target:",
            "primitive-local DeltaPhi in [0, 1].",
        ]
    )


def primitive_chain_from_row(
    row: Mapping[str, Any],
    max_primitives: int = DEFAULT_MAX_PRIMITIVES,
) -> tuple[str, ...]:
    chain: list[str] = []
    for index in range(1, max_primitives + 1):
        primitive = clean_text(row.get(f"primitive_{index}"))
        obj = clean_text(row.get(f"object_{index}"))
        if not primitive and not obj:
            continue
        chain.append(" ".join(part for part in (primitive, obj) if part))
    return tuple(chain)


def prompt_records_from_rows(
    rows: Iterable[Mapping[str, Any]],
    max_primitives: int = DEFAULT_MAX_PRIMITIVES,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    seen: set[str] = set()
    for row in rows:
        task_id = clean_text(row.get("task_id"))
        if not task_id:
            continue
        if task_id in seen:
            raise ValueError(f"Duplicate task_id in prompt table: {task_id}")
        task_meta_text = clean_text(row.get("task_meta_text"))
        if not task_meta_text:
            raise ValueError(f"Missing task_meta_text for task_id={task_id}")
        primitive_chain = primitive_chain_from_row(row, max_primitives=max_primitives)
        records.append(
            PromptRecord(
                task_id=task_id,
                task_meta_text=task_meta_text,
                primitive_chain=primitive_chain,
                prompt=format_prompt(task_meta_text, primitive_chain),
            )
        )
        seen.add(task_id)
    if not records:
        raise ValueError("No prompt records were parsed.")
    return records


def _xlsx_col_index(cell_ref: str) -> int:
    match = CELL_RE.match(cell_ref)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("x:si", SPREADSHEET_NS):
        values.append("".join(text.text or "" for text in item.findall(".//x:t", SPREADSHEET_NS)))
    return values


def _sheet_path(zf: ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_by_id = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("r:Relationship", REL_NS)}
    for sheet in workbook.findall(".//x:sheet", SPREADSHEET_NS):
        if sheet.attrib.get("name") != sheet_name:
            continue
        rel_id = sheet.attrib[f"{{{OFFICE_REL}}}id"]
        target = rel_by_id[rel_id]
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"Sheet {sheet_name!r} not found in workbook.")


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//x:t", SPREADSHEET_NS))

    value = cell.find("x:v", SPREADSHEET_NS)
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def read_xlsx_sheet_rows(path: str | Path, sheet_name: str = MAIN_SHEET) -> list[dict[str, str]]:
    """Read an XLSX worksheet as row dictionaries without requiring openpyxl."""
    path = Path(path)
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_xml = ET.fromstring(zf.read(_sheet_path(zf, sheet_name)))
        matrix: list[list[str]] = []
        for row in sheet_xml.findall(".//x:sheetData/x:row", SPREADSHEET_NS):
            values: list[str] = []
            for cell in row.findall("x:c", SPREADSHEET_NS):
                ref = cell.attrib.get("r", "")
                column = _xlsx_col_index(ref)
                while len(values) <= column:
                    values.append("")
                values[column] = _cell_value(cell, shared_strings)
            matrix.append(values)

    if not matrix:
        raise ValueError(f"No rows found in {path}:{sheet_name}.")
    headers = [clean_text(value) for value in matrix[0]]
    rows: list[dict[str, str]] = []
    for values in matrix[1:]:
        if not any(clean_text(value) for value in values):
            continue
        row = {header: values[index] if index < len(values) else "" for index, header in enumerate(headers) if header}
        rows.append(row)
    return rows


def read_prompt_records(
    excel_path: str | Path,
    sheet_name: str = MAIN_SHEET,
    max_primitives: int = DEFAULT_MAX_PRIMITIVES,
) -> list[PromptRecord]:
    rows = read_xlsx_sheet_rows(excel_path, sheet_name=sheet_name)
    return prompt_records_from_rows(rows, max_primitives=max_primitives)


def write_prompt_table(records: Iterable[PromptRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            payload["primitive_chain"] = list(record.primitive_chain)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_prompt_table(path: str | Path) -> list[PromptRecord]:
    path = Path(path)
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                PromptRecord(
                    task_id=str(payload["task_id"]),
                    task_meta_text=str(payload["task_meta_text"]),
                    primitive_chain=tuple(str(item) for item in payload["primitive_chain"]),
                    prompt=str(payload["prompt"]),
                )
            )
    if not records:
        raise ValueError(f"No prompt records found in {path}.")
    return records


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def encode_prompts_mock(records: Iterable[PromptRecord], feature_dim: int, seed: int = 42) -> np.ndarray:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    features: list[np.ndarray] = []
    for record in records:
        rng = np.random.default_rng(_stable_seed(seed, record.task_id, record.prompt))
        vector = rng.normal(size=(feature_dim,)).astype(np.float32)
        norm = np.linalg.norm(vector)
        features.append(vector / max(float(norm), 1e-6))
    return np.stack(features, axis=0).astype(np.float32)


def _model_text_features(model: torch.nn.Module, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        try:
            return model.get_text_features(**encoded)
        except TypeError:
            filtered = {key: value for key, value in encoded.items() if key != "token_type_ids"}
            return model.get_text_features(**filtered)

    output = model(**encoded)
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    text_embeds = getattr(output, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)):
        hidden = output[0]
    if hidden is None:
        raise RuntimeError("Could not find text features in transformer model output.")
    mask = encoded.get("attention_mask")
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_prompts_transformers(
    records: Iterable[PromptRecord],
    model_name: str = DEFAULT_TEXT_MODEL,
    batch_size: int = 32,
    device_name: str | None = None,
) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required for SigLIP/CLIP prompt encoding.") from exc

    records = list(records)
    if not records:
        raise ValueError("records must not be empty.")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            encoded = tokenizer(
                [record.prompt for record in batch_records],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            features.append(_model_text_features(model, encoded).detach().cpu().float())
    return torch.cat(features, dim=0).numpy().astype(np.float32)


def write_prompt_feature_store(
    path: str | Path,
    records: Iterable[PromptRecord],
    features: np.ndarray,
) -> None:
    records = list(records)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape [num_tasks, prompt_dim].")
    if features.shape[0] != len(records):
        raise ValueError("features row count must match records.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        task_ids=np.asarray([record.task_id for record in records]),
        features=features,
    )


def load_prompt_feature_store(
    path: str | Path,
    expected_dim: int | None = None,
) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        if "task_ids" not in loaded or "features" not in loaded:
            raise ValueError(f"{path} must contain task_ids and features arrays.")
        task_ids = [str(task_id) for task_id in loaded["task_ids"].tolist()]
        features = loaded["features"].astype(np.float32)
    if features.ndim != 2:
        raise ValueError("prompt features must have shape [num_tasks, prompt_dim].")
    if len(task_ids) != features.shape[0]:
        raise ValueError("task_ids length must match prompt feature rows.")
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(f"Prompt features have dim {features.shape[1]}; expected {expected_dim}.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("prompt feature store contains duplicate task_ids.")
    return {task_id: features[index] for index, task_id in enumerate(task_ids)}


def prepare_prompt_artifacts(
    excel_path: str | Path,
    output_dir: str | Path,
    sheet_name: str = MAIN_SHEET,
    encoder: str = "transformers",
    model_name: str = DEFAULT_TEXT_MODEL,
    feature_dim: int = 512,
    batch_size: int = 32,
    device_name: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    records = read_prompt_records(excel_path, sheet_name=sheet_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "prompt_table.jsonl"
    feature_path = output_dir / "prompt_features.npz"
    manifest_path = output_dir / "prompt_manifest.json"

    if encoder == "mock":
        features = encode_prompts_mock(records, feature_dim=feature_dim, seed=seed)
    elif encoder == "transformers":
        features = encode_prompts_transformers(
            records,
            model_name=model_name,
            batch_size=batch_size,
            device_name=device_name,
        )
        feature_dim = int(features.shape[1])
    else:
        raise ValueError(f"Unsupported prompt encoder: {encoder}")

    write_prompt_table(records, table_path)
    write_prompt_feature_store(feature_path, records, features)
    manifest = {
        "source_excel": str(excel_path),
        "sheet_name": sheet_name,
        "encoder": encoder,
        "model_name": model_name if encoder == "transformers" else "mock",
        "feature_dim": feature_dim,
        "num_prompts": len(records),
        "prompt_table": str(table_path),
        "prompt_features": str(feature_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GM-100 task prompts and frozen text features.")
    parser.add_argument("--excel", required=True, help="Path to primitive_chain_gm100.xlsx.")
    parser.add_argument("--output", required=True, help="Output directory for prompt table/features.")
    parser.add_argument("--sheet", default=MAIN_SHEET)
    parser.add_argument("--encoder", choices=("transformers", "mock"), default="transformers")
    parser.add_argument("--model", default=DEFAULT_TEXT_MODEL, help="Hugging Face SigLIP/CLIP model name.")
    parser.add_argument("--feature-dim", type=int, default=512, help="Mock feature dimension.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = prepare_prompt_artifacts(
            excel_path=args.excel,
            output_dir=args.output,
            sheet_name=args.sheet,
            encoder=args.encoder,
            model_name=args.model,
            feature_dim=args.feature_dim,
            batch_size=args.batch_size,
            device_name=args.device,
            seed=args.seed,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
