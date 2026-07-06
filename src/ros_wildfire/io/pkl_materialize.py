from __future__ import annotations

import gzip
import pickle
import shutil
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple, Any, Iterable


def load_pickle_dataset(pkl_path: str) -> Dict[str, Any]:
    p = Path(pkl_path)
    if not p.exists():
        raise FileNotFoundError(f"PKL not found: {p}")
    suf = "".join(p.suffixes).lower()

    if suf.endswith(".pkl.gz") or suf.endswith(".pickle.gz"):
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    if suf.endswith(".pkl") or suf.endswith(".pickle"):
        with open(p, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported pickle suffix: {p.suffixes}")


def _norm_rel(rel: str) -> str:
    return str(rel).replace("\\", "/").lstrip("/")


def _write_bytes_map(root: Path, bytes_map: Dict[str, bytes]) -> None:
    for rel, blob in bytes_map.items():
        rel = _norm_rel(rel)
        out_path = root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)


_SEQ_RE = re.compile(r"^(?P<case>\d{4})_(?P<seq>\d{5})\.txt$")


def _discover_idxs_from_seq_bytes(seq_bytes_map: Dict[str, bytes], case_id: str) -> List[int]:
    idxs: List[int] = []
    if not seq_bytes_map:
        return idxs
    for k in seq_bytes_map.keys():
        kk = _norm_rel(k)
        if kk.startswith("Weather_Data/") and kk.endswith(".txt"):
            name = kk.split("/")[-1]
            m = _SEQ_RE.match(name)
            if m and m.group("case") == case_id:
                idxs.append(int(m.group("seq")))
    return sorted(set(idxs))


def _select_idxs(
    all_idxs: Iterable[int],
    max_n: int | None,
    mode: str | None,
    seed: int | None,
    seq_list: Iterable[int] | None,
) -> List[int]:
    all_idxs = sorted(int(x) for x in all_idxs)
    mode = (mode or "first").lower()

    if mode == "list":
        if not seq_list:
            return all_idxs if max_n is None else all_idxs[: int(max_n)]
        keep = [int(x) for x in seq_list if int(x) in set(all_idxs)]
        return keep if max_n is None else keep[: int(max_n)]

    if max_n is None:
        return all_idxs

    max_n = int(max_n)
    if mode == "random":
        rng = random.Random(int(seed or 0))
        if len(all_idxs) <= max_n:
            return all_idxs
        return sorted(rng.sample(all_idxs, k=max_n))

    return all_idxs[:max_n]


def _filter_sequence_bytes_map(
    seq_bytes_map: Dict[str, bytes],
    case_id: str,
    keep_idxs: Iterable[int],
) -> Dict[str, bytes]:
    keep_set = set(int(x) for x in keep_idxs)
    out: Dict[str, bytes] = {}

    for rel, blob in (seq_bytes_map or {}).items():
        r = _norm_rel(rel)

        if r.startswith("Weather_Data/") and r.endswith(".txt"):
            name = r.split("/")[-1]
            m = _SEQ_RE.match(name)
            if m and m.group("case") == case_id and int(m.group("seq")) in keep_set:
                out[rel] = blob
            continue

        if r.startswith(("Satellite_Images_Mask/", "Satellite_Image_Mask/")):
            parts = r.split("/")
            if len(parts) >= 2:
                folder = parts[1]
                m = re.match(rf"^{case_id}_(\d{{5}})$", folder)
                if m and int(m.group(1)) in keep_set:
                    out[rel] = blob
            continue

    return out


def materialize_one_case(
    case_obj: Dict,
    *,
    out_base: Path,
    case_index: int,
    max_seqs: int | None = None,
    seq_sample_mode: str | None = None,
    seq_list: Iterable[int] | None = None,
    seq_seed: int | None = None,
    reset: bool = False,
) -> Tuple[Path, Path]:
    """
    Materialize one PKL 'case' into:
      scenario_dir = out_base / f"{case_id}_{case_index:03d}_pkl"
      case_root    = scenario_dir / case_id
    """
    case_id = case_obj["case_id"]
    scenario_dir = out_base / f"{case_id}_{case_index:05d}_pkl"
    case_root = scenario_dir / case_id

    if reset and scenario_dir.exists():
        shutil.rmtree(scenario_dir, ignore_errors=True)

    case_root.mkdir(parents=True, exist_ok=True)

    static_map = case_obj.get("files", {}).get("static", {}) or {}
    seq_map = case_obj.get("files", {}).get("sequence", {}) or {}

    _write_bytes_map(case_root, static_map)

    if max_seqs is None and (seq_sample_mode or "first").lower() != "list":
        _write_bytes_map(case_root, seq_map)
    else:
        all_idxs = _discover_idxs_from_seq_bytes(seq_map, case_id)
        keep_idxs = _select_idxs(all_idxs, max_seqs, seq_sample_mode, seq_seed, seq_list)
        seq_map_filt = _filter_sequence_bytes_map(seq_map, case_id, keep_idxs)
        _write_bytes_map(case_root, seq_map_filt)
    return scenario_dir, case_root


def materialize_all_cases(
    dataset: Dict,
    *,
    out_base: Path,
    max_cases: int | None = None,
    case_filter: Iterable[str] | None = None,
    max_seqs_per_case: int | None = None,
    seq_sample_mode: str | None = None,
    seq_list: Iterable[int] | None = None,
    seq_seed: int | None = None,
    reset: bool = False,
) -> List[str]:
    cases = dataset.get("cases", [])
    if not cases:
        raise RuntimeError("Dataset has no 'cases'.")

    out_base.mkdir(parents=True, exist_ok=True)
    raw_case_filter = [str(x).strip() for x in (case_filter or []) if str(x).strip()]

    def _case_allowed(case_id: str) -> bool:
        if not raw_case_filter:
            return True
        cid = str(case_id)
        return any(cid.startswith(f) for f in raw_case_filter)

    selected = [i for i, c in enumerate(cases) if _case_allowed(c.get("case_id", ""))]
    if max_cases is not None:
        selected = selected[: int(max_cases)]

    if not selected:
        sample = [c.get("case_id", "") for c in cases[:10]]
        raise RuntimeError(
            f"case_filter={raw_case_filter} produced no matches. Example case_ids: {sample}"
        )

    scenario_dirs: List[str] = []
    for n, i in enumerate(selected, start=1):
        print(f"Materializing case {n}/{len(selected)} (case_id={cases[i]['case_id']})...")
        scenario_dir, _ = materialize_one_case(
            cases[i],
            out_base=out_base,
            case_index=i,
            max_seqs=max_seqs_per_case,
            seq_sample_mode=seq_sample_mode,
            seq_list=seq_list,
            seq_seed=seq_seed,
            reset=reset,
        )
        scenario_dirs.append(str(scenario_dir))
    return scenario_dirs
