#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
#########################################################################
"""Instance-level cache helpers for warm-start artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Optional

import arguments


def _stable_json_dumps(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_file_id(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_").replace(":", "_")


def build_model_signature() -> str:
    model_cfg = arguments.Config["model"]
    general_cfg = arguments.Config["general"]
    solver_cfg = arguments.Config["solver"]
    bab_cfg = arguments.Config["bab"]
    spec_cfg = arguments.Config["specification"]
    payload = {
        "name": model_cfg.get("name", ""),
        "path": model_cfg.get("path", ""),
        "onnx_path": model_cfg.get("onnx_path", ""),
        "input_shape": model_cfg.get("input_shape", None),
        "device": general_cfg.get("device", ""),
        "complete_verifier": general_cfg.get("complete_verifier", ""),
        "bound_prop_method": solver_cfg.get("bound_prop_method", ""),
        "norm": str(spec_cfg.get("norm", "")),
        "epsilon": spec_cfg.get("epsilon", None),
        "branching_method": bab_cfg.get("branching", {}).get("method", ""),
    }
    return _sha256_text(_stable_json_dumps(payload))


def build_spec_signature(vnnlib_id: int, vnnlib_handler=None) -> str:
    spec_cfg = arguments.Config["specification"]
    data_cfg = arguments.Config["data"]
    payload: dict[str, Any] = {
        "type": spec_cfg.get("type", ""),
        "robustness_type": spec_cfg.get("robustness_type", ""),
        "norm": str(spec_cfg.get("norm", "")),
        "epsilon": spec_cfg.get("epsilon", None),
        "dataset": data_cfg.get("dataset", ""),
    }
    if vnnlib_handler is not None:
        input_shape = getattr(vnnlib_handler, "input_shape", None)
        num_output = getattr(vnnlib_handler, "num_output", None)
        total_num_or = getattr(vnnlib_handler, "total_num_or", None)
        or_spec_size = getattr(vnnlib_handler, "or_spec_size", None)
        c = getattr(vnnlib_handler, "c", None)
        rhs = getattr(vnnlib_handler, "rhs", None)
        signature_probe = {
            "input_shape": input_shape,
            "num_output": num_output,
            "total_num_or": int(total_num_or) if total_num_or is not None else None,
            "or_spec_size_shape": list(or_spec_size.shape) if hasattr(or_spec_size, "shape") else None,
            "or_spec_size_sum": int(or_spec_size.sum().item()) if hasattr(or_spec_size, "sum") else None,
            "c_shape": list(c.shape) if hasattr(c, "shape") else None,
            "rhs_shape": list(rhs.shape) if hasattr(rhs, "shape") else None,
        }
        if c is not None and rhs is not None:
            c_sample = c.detach().flatten()[:128].cpu().tolist()
            rhs_sample = rhs.detach().flatten()[:128].cpu().tolist()
            signature_probe["probe_hash"] = _sha256_text(
                _stable_json_dumps({"c_sample": c_sample, "rhs_sample": rhs_sample})
            )
        payload.update(signature_probe)
    return _sha256_text(_stable_json_dumps(payload))


@dataclass
class CacheEntryMeta:
    entry_id: str
    model_sig: str
    spec_sig: str
    created_at: float
    last_used_at: float
    hit_count: int = 0
    avg_time_sec: Optional[float] = None
    avg_domains_visited: Optional[float] = None
    branching_method: Optional[str] = None


class InstanceCacheManager:
    def __init__(self) -> None:
        cache_cfg = arguments.Config["cache"]
        self.enabled = bool(cache_cfg.get("enabled", False))
        self.alpha_warmstart = bool(cache_cfg.get("alpha_warmstart", False))
        self.branching_hints = bool(cache_cfg.get("branching_hints", False))
        self.max_entries = int(cache_cfg.get("max_entries", 200))
        self.path = cache_cfg.get("path", "") or ""
        self._index_path: Optional[str] = None
        self._entries_dir: Optional[str] = None
        self._index: dict[str, dict[str, Any]] = {"entries": {}}
        self.last_lookup_hit: bool = False
        self.last_lookup_exact: bool = False

        if not self.enabled:
            return
        if not self.path:
            self.path = os.path.join(os.getcwd(), ".abcrown_instance_cache")
        os.makedirs(self.path, exist_ok=True)
        self._entries_dir = os.path.join(self.path, "entries")
        os.makedirs(self._entries_dir, exist_ok=True)
        self._index_path = os.path.join(self.path, "index.json")
        self._load_index()

    def _load_index(self) -> None:
        if not self.enabled or self._index_path is None:
            return
        if not os.path.exists(self._index_path):
            self._save_index()
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        except Exception:
            self._index = {"entries": {}}
            self._save_index()

    def _save_index(self) -> None:
        if not self.enabled or self._index_path is None:
            return
        tmp = f"{self._index_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, sort_keys=True)
        os.replace(tmp, self._index_path)

    def _entry_path(self, entry_id: str) -> str:
        assert self._entries_dir is not None
        return os.path.join(self._entries_dir, f"{_safe_file_id(entry_id)}.pkl")

    def _evict_if_needed(self) -> None:
        entries = self._index.get("entries", {})
        if len(entries) <= self.max_entries:
            return
        ordered = sorted(entries.items(), key=lambda kv: kv[1].get("last_used_at", 0.0))
        to_remove = len(entries) - self.max_entries
        for entry_id, _meta in ordered[:to_remove]:
            path = self._entry_path(entry_id)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            entries.pop(entry_id, None)

    def _find_best_entry_id(self, model_sig: str, spec_sig: str) -> tuple[Optional[str], bool]:
        entries = self._index.get("entries", {})
        exact = [
            (eid, m) for eid, m in entries.items()
            if m.get("model_sig") == model_sig and m.get("spec_sig") == spec_sig
        ]
        if exact:
            exact.sort(key=lambda kv: kv[1].get("last_used_at", 0.0), reverse=True)
            return exact[0][0], True
        same_model = [
            (eid, m) for eid, m in entries.items()
            if m.get("model_sig") == model_sig
        ]
        if same_model:
            same_model.sort(key=lambda kv: kv[1].get("last_used_at", 0.0), reverse=True)
            return same_model[0][0], False
        return None, False

    def load_for_instance(self, model_sig: str, spec_sig: str) -> dict[str, Any]:
        self.last_lookup_hit = False
        self.last_lookup_exact = False
        if not self.enabled:
            return {}
        entry_id, exact_match = self._find_best_entry_id(model_sig, spec_sig)
        if entry_id is None:
            return {}
        path = self._entry_path(entry_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return {}
        now = time.time()
        meta = self._index["entries"].get(entry_id, {})
        meta["last_used_at"] = now
        meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
        self._index["entries"][entry_id] = meta
        self._save_index()
        self.last_lookup_hit = True
        self.last_lookup_exact = exact_match
        return payload if isinstance(payload, dict) else {}

    def save_for_instance(
        self,
        model_sig: str,
        spec_sig: str,
        *,
        alpha: Any = None,
        instance_time_sec: Optional[float] = None,
        domains_visited: Optional[float] = None,
        branching_method: Optional[str] = None,
        branching_layer_hist: Optional[dict[str, int]] = None,
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        entry_id = _sha256_text(f"{model_sig}:{spec_sig}")
        prev_payload = {}
        path = self._entry_path(entry_id)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    prev_payload = pickle.load(f)
            except Exception:
                prev_payload = {}
        preserved_alpha = prev_payload.get("alpha", None) if isinstance(prev_payload, dict) else None
        payload = {
            "model_sig": model_sig,
            "spec_sig": spec_sig,
            "alpha": alpha if alpha is not None else preserved_alpha,
            "instance_time_sec": instance_time_sec,
            "domains_visited": domains_visited,
            "branching_method": branching_method,
            "branching_layer_hist": branching_layer_hist or {},
            "saved_at": now,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        old = self._index["entries"].get(entry_id, {})
        # Lightweight running averages.
        old_n = int(old.get("sample_count", 0))
        new_n = old_n + 1
        old_t = old.get("avg_time_sec")
        old_d = old.get("avg_domains_visited")
        if instance_time_sec is not None:
            avg_time = float(instance_time_sec) if old_t is None else ((old_t * old_n) + float(instance_time_sec)) / new_n
        else:
            avg_time = old_t
        if domains_visited is not None:
            avg_dom = float(domains_visited) if old_d is None else ((old_d * old_n) + float(domains_visited)) / new_n
        else:
            avg_dom = old_d

        self._index["entries"][entry_id] = {
            "entry_id": entry_id,
            "model_sig": model_sig,
            "spec_sig": spec_sig,
            "created_at": old.get("created_at", now),
            "last_used_at": now,
            "hit_count": int(old.get("hit_count", 0)),
            "sample_count": new_n,
            "avg_time_sec": avg_time,
            "avg_domains_visited": avg_dom,
            "branching_method": branching_method or old.get("branching_method"),
        }
        self._evict_if_needed()
        self._save_index()

    def suggest_branching_tuning(self, model_sig: Optional[str] = None) -> dict[str, Any]:
        """Return conservative branching hint knobs from historical cache stats."""
        if not (self.enabled and self.branching_hints):
            return {}
        entries = list(self._index.get("entries", {}).values())
        if model_sig is not None:
            entries = [e for e in entries if e.get("model_sig") == model_sig]
        if not entries:
            return {}
        # Use model-level aggregate over all entries.
        avg_domains = [
            e.get("avg_domains_visited") for e in entries
            if e.get("avg_domains_visited") is not None
        ]
        if not avg_domains:
            return {}
        mean_domains = sum(avg_domains) / len(avg_domains)
        # Conservative hint: if historically hard, increase candidates modestly.
        if mean_domains > 500:
            return {"candidates_delta": 1}
        return {}
