import copy
import hashlib
import random
import torch

import nnverify.domains
from nnverify import config
from nnverify.bnb import Split, is_relu_split, is_input_split
from nnverify.common import Status
from nnverify.domains.deepz import ZonoTransformer
from nnverify.specs.spec import Spec, SpecList


def branch_unsolved(spec_list, split, split_score=None, inp_template=None, args=None, net=None, transformer=None,
                    branch_observer=None, candidate_limit=5):
    new_spec_list = SpecList()
    verified_specs = SpecList()

    for spec in spec_list:
        if spec.status == Status.UNKNOWN:
            add_spec = split_spec(spec, split, split_score=split_score,
                                  inp_template=inp_template,
                                  args=args, net=net, transformer=transformer,
                                  branch_observer=branch_observer, candidate_limit=candidate_limit)
            new_spec_list += SpecList(add_spec)
        else:
            verified_specs.append(spec)
    return new_spec_list, verified_specs


def split_spec(spec, split_type, split_score=None, inp_template=None, args=None, net=None, transformer=None,
               branch_observer=None, candidate_limit=5):
    if is_relu_split(split_type):
        spec.chosen_split = choose_relu(split_type, spec.relu_spec, spec=spec, split_score=split_score,
                                        inp_template=inp_template, args=args, transformer=transformer)
        split_relu_specs = spec.relu_spec.split_spec(split_type, spec.chosen_split)
        child_specs = [Spec(spec.input_spec, rs, parent=spec) for rs in split_relu_specs]
    elif is_input_split(split_type):
        spec.chosen_split = choose_split_dim(spec.input_spec, split_type, net=net)
        split_inp_specs = spec.input_spec.split_spec(split_type, spec.chosen_split)
        child_specs = [Spec(ins, spec.relu_spec, parent=spec) for ins in split_inp_specs]
    else:
        raise ValueError("Unknown split!")
    spec.children += child_specs
    if branch_observer is not None:
        branch_observer(
            {
                "node_lower_bound": _scalar_lb(spec.lb),
                "node_depth": get_spec_depth(spec),
                "node_signature": summarize_node_state(spec),
                "chosen_split": serialize_split_identifier(spec.chosen_split),
                "chosen_split_score": get_chosen_split_score(spec.chosen_split, split_score),
                "top_candidates": get_top_candidates(spec, split_type, split_score=split_score, limit=candidate_limit),
                "candidate_ranking": get_candidate_ranking_summary(
                    spec,
                    split_type,
                    chosen_split=spec.chosen_split,
                    split_score=split_score,
                ),
                "child_count": len(child_specs),
            }
        )
    return child_specs


def get_spec_depth(spec):
    depth = 0
    cur_spec = spec.parent
    while cur_spec is not None:
        depth += 1
        cur_spec = cur_spec.parent
    return depth


def _scalar_lb(lb):
    if lb is None:
        return None
    if isinstance(lb, torch.Tensor):
        lb = lb.detach().cpu()
        if lb.numel() == 1:
            return float(lb.item())
        return [float(v) for v in lb.flatten().tolist()]
    return lb


def serialize_split_identifier(split_identifier):
    if split_identifier is None:
        return None
    if isinstance(split_identifier, tuple) and len(split_identifier) == 2:
        return {"kind": "relu", "layer": int(split_identifier[0]), "neuron": int(split_identifier[1])}
    return {"kind": "input", "index": int(split_identifier)}


def digest_relu_mask(relu_mask):
    if relu_mask is None:
        return None
    digest = hashlib.sha1()
    for relu_id, decision in sorted(relu_mask.items()):
        digest.update(str(relu_id).encode("utf-8"))
        digest.update(str(int(decision)).encode("utf-8"))
    return digest.hexdigest()[:12]


def summarize_relu_mask(relu_mask):
    if relu_mask is None:
        return None

    per_layer = {}
    active = 0
    passive = 0
    ambiguous = 0
    for relu_id, decision in relu_mask.items():
        layer_idx = int(relu_id[0])
        stats = per_layer.setdefault(layer_idx, {"layer_index": layer_idx, "active": 0, "passive": 0, "ambiguous": 0})
        if decision == 1:
            active += 1
            stats["active"] += 1
        elif decision == -1:
            passive += 1
            stats["passive"] += 1
        else:
            ambiguous += 1
            stats["ambiguous"] += 1

    return {
        "mask_digest": digest_relu_mask(relu_mask),
        "total_relus": len(relu_mask),
        "assigned_relus": active + passive,
        "active_relus": active,
        "passive_relus": passive,
        "ambiguous_relus": ambiguous,
        "per_layer": [per_layer[layer_idx] for layer_idx in sorted(per_layer.keys())],
    }


def summarize_input_spec(input_spec):
    if input_spec is None:
        return None
    widths = input_spec.input_ub - input_spec.input_lb
    widths = widths.detach().cpu().flatten()
    return {
        "input_width_digest": hashlib.sha1(widths.numpy().tobytes()).hexdigest()[:12],
        "input_dim": int(widths.numel()),
        "max_width": float(torch.max(widths).item()),
        "mean_width": float(torch.mean(widths.float()).item()),
        "nonzero_width_dims": int(torch.count_nonzero(widths > 0).item()),
    }


def summarize_node_state(spec):
    return {
        "depth": get_spec_depth(spec),
        "relu_mask": summarize_relu_mask(spec.relu_spec.relu_mask) if spec.relu_spec is not None else None,
        "input_region": summarize_input_spec(spec.input_spec),
    }


def get_chosen_split_score(chosen_split, split_score):
    if split_score is None or chosen_split is None:
        return None
    score = split_score.get(chosen_split)
    if score is None:
        return None
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu()
        if score.numel() == 1:
            return float(score.item())
        return [float(v) for v in score.flatten().tolist()]
    return float(score)


def _sortable_score(score):
    if score is None:
        return float("-inf")
    if isinstance(score, list):
        return score[0] if len(score) > 0 else float("-inf")
    return float(score)


def build_candidate_entries(spec, split_type, split_score=None):
    if is_relu_split(split_type):
        relu_mask = spec.relu_spec.relu_mask
        candidates = []
        for relu, decision in relu_mask.items():
            if decision != 0:
                continue
            entry = {"split": serialize_split_identifier(relu), "raw_split": relu}
            if split_score is not None and relu in split_score:
                score = split_score[relu]
                if isinstance(score, torch.Tensor):
                    score = float(score.detach().cpu().item()) if score.numel() == 1 else [
                        float(v) for v in score.detach().cpu().flatten().tolist()
                    ]
                entry["score"] = score
            else:
                entry["score"] = None
            candidates.append(entry)
        candidates.sort(key=lambda item: _sortable_score(item["score"]), reverse=True)
        return candidates

    if is_input_split(split_type):
        widths = spec.input_spec.input_ub - spec.input_spec.input_lb
        candidates = []
        for index, value in enumerate(widths.flatten()):
            candidates.append(
                {
                    "split": serialize_split_identifier(index),
                    "raw_split": index,
                    "score": float(value.item()),
                }
            )
        candidates.sort(key=lambda item: _sortable_score(item["score"]), reverse=True)
        return candidates

    return []


def get_top_candidates(spec, split_type, split_score=None, limit=5):
    candidates = build_candidate_entries(spec, split_type, split_score=split_score)
    return [{"split": candidate["split"], "score": candidate["score"]} for candidate in candidates[:limit]]


def get_candidate_ranking_summary(spec, split_type, chosen_split, split_score=None):
    candidates = build_candidate_entries(spec, split_type, split_score=split_score)
    if len(candidates) == 0:
        return {"candidate_count": 0, "chosen_rank": None, "top1_top2_gap": None, "chosen_gap_from_top": None}

    chosen_rank = None
    chosen_score = None
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["raw_split"] == chosen_split:
            chosen_rank = rank
            chosen_score = candidate["score"]
            break

    top_score = candidates[0]["score"]
    second_score = candidates[1]["score"] if len(candidates) > 1 else None
    return {
        "candidate_count": len(candidates),
        "chosen_rank": chosen_rank,
        "top1_top2_gap": None if second_score is None else _sortable_score(top_score) - _sortable_score(second_score),
        "chosen_gap_from_top": None if chosen_score is None else _sortable_score(top_score) - _sortable_score(chosen_score),
    }


def choose_relu(split, relu_spec, spec=None, split_score=None, inp_template=None, args=None, transformer=None):
    """
    Chooses the relu that is split in branch and bound.
    @param: relu_spec contains relu_mask which is a map that maps relus to -1/0/1. 0 here indicates that the relu
        is ambiguous
    """
    relu_mask = relu_spec.relu_mask
    if split == Split.RELU_RAND:
        all_relus = []

        # Collect all un-split relus
        for relu in relu_mask.keys():
            if relu_mask[relu] == 0 and relu[0] == 2:
                all_relus.append(relu)

        return random.choice(all_relus)

    # BaBSR based on various estimates of importance
    elif split == Split.RELU_GRAD or split == Split.RELU_ESIP_SCORE or split == Split.RELU_ESIP_SCORE2:
        # Choose the ambiguous relu that has the maximum score in relu_score
        if split_score is None:
            raise ValueError("relu_score should be set while using relu_grad splitting mode")

        max_score, chosen_relu = 0, None

        for relu in relu_mask.keys():
            if relu_mask[relu] == 0 and relu in split_score.keys():
                if split_score[relu] >= max_score:
                    max_score, chosen_relu = split_score[relu], relu

        if chosen_relu is None:
            raise ValueError("Attempt to split should only take place if there are ambiguous relus!")

        config.write_log("Chosen relu for splitting: " + str(chosen_relu) + " " + str(max_score))
        return chosen_relu
    elif split == Split.RELU_KFSB:
        k = 3
        if split_score is None:
            raise ValueError("relu_score should be set while using kFSB splitting mode")
        if spec is None:
            raise ValueError("spec should be set while using kFSB splitting mode")

        candidate_relu_score_list = []
        for relu in relu_mask.keys():
            if relu_mask[relu] == 0 and relu in split_score.keys():
                candidate_relu_score_list.append((relu, split_score[relu]))
        candidate_relu_score_list = sorted(candidate_relu_score_list, key=lambda x: x[1], reverse=True)
        candidate_relus = [candidate_relu_score_list[i][0] for i in range(k)]

        candidate_relu_lbs = {}
        for relu in candidate_relus:
            cp_spec = copy.deepcopy(spec)
            split_relu_specs = cp_spec.relu_spec.split_spec(split, relu)
            child_specs = [Spec(cp_spec.input_spec, rs, parent=cp_spec) for rs in split_relu_specs]

            candidate_lb = 0
            for child_spec in child_specs:
                transformer.update_spec(child_spec.input_spec, relu_mask=child_spec.relu_spec.relu_mask)
                lb, _, _ = transformer.compute_lb(complete=True)
                if lb is not None:
                    candidate_lb = min(candidate_lb, lb)

            candidate_relu_lbs[relu] = candidate_lb
        return max(candidate_relu_lbs, key=candidate_relu_lbs.get)
    else:
        # Returning just the first un-split relu
        for relu in relu_mask.keys():
            if relu_mask[relu] == 0:
                return relu
    raise ValueError("No relu chosen!")


def choose_split_dim(input_spec, split, net=None):
    if split == Split.INPUT:
        chosen_dim = torch.argmax(input_spec.input_ub - input_spec.input_lb)
    elif split == Split.INPUT_GRAD:
        zono_transformer = ZonoTransformer(input_spec, complete=True)
        zono_transformer = nnverify.domains.build_transformer(zono_transformer, net, input_spec)

        center = zono_transformer.centers[-1]
        cof = zono_transformer.cofs[-1]
        cof_abs = torch.sum(torch.abs(cof), dim=0)
        lb = center - cof_abs

        if input_spec.out_constr.is_conjunctive:
            adv_index = torch.argmin(lb)
        else:
            adv_index = torch.argmax(lb)

        input_len = len(input_spec.input_lb)
        chosen_noise_idx = torch.argmax(torch.abs(cof[:input_len, adv_index])).item()
        # chosen_noise_idx = torch.argmax(torch.sum(torch.abs(cof[:input_len]), dim=1) * (self.input_ub - self.input_lb))
        chosen_dim = zono_transformer.map_for_noise_indices[chosen_noise_idx]
    elif split == Split.INPUT_SB:
        cp_spec = copy.deepcopy(input_spec)
        lb0 = input_spec.get_zono_lb(net, cp_spec)

        chosen_dim = -1
        best_score = -1e-3

        for dim in range(len(input_spec.input_lb)):
            s1, s2 = cp_spec.split_spec(split, dim)

            lb1 = input_spec.get_zono_lb(net, s1)
            lb2 = input_spec.get_zono_lb(net, s2)

            dim_score = min(lb1 - lb0, lb2 - lb0)

            if dim_score > best_score:
                chosen_dim = dim
                best_score = dim_score
    else:
        raise ValueError("Unknown splitting method!")
    return chosen_dim


def split_chosen_spec(spec, split_type, chosen_split):
    spec.chosen_split = chosen_split
    if is_relu_split(split_type):
        split_relu_specs = spec.relu_spec.split_spec(split_type, chosen_split)
        child_specs = [Spec(spec.input_spec, rs, parent=spec) for rs in split_relu_specs]
    elif is_input_split(split_type):
        split_inp_specs = spec.input_spec.split_spec(split_type, chosen_split)
        child_specs = [Spec(ins, spec.relu_spec, parent=spec) for ins in split_inp_specs]
    else:
        raise ValueError("Unknown split!")
    spec.children += child_specs
    return child_specs
