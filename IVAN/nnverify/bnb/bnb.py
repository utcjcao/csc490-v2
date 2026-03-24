"""
A generic approach for the BnB based complete verification.
"""
import torch
import nnverify.domains
import nnverify.specs.spec as specs
import time

from nnverify.domains import get_domain_transformer
from nnverify import config
from nnverify.bnb import Split, is_relu_split, branch
from nnverify.bnb.proof_tree import ProofTree
from nnverify.domains.deepz import ZonoTransformer
from nnverify.common import Status
from nnverify.proof_transfer.pt_types import ProofTransferMethod, IVAN, REORDERING
from multiprocessing import Pool

from nnverify.specs.input_spec import merge_input_specs


class BnB:
    def __init__(self, net, transformer, init_prop, args, template_store=None, print_result=False,
                 artifact_logger=None):
        self.net = net
        self.transformer = transformer
        self.init_prop = init_prop
        self.split = args.split
        self.template_store = template_store
        self.args = args
        self.depth = 1
        self.init_time = time.time()
        self.global_status = Status.UNKNOWN
        self.print_result = print_result
        self.artifact_logger = artifact_logger

        # Store proof tree for the BnB
        self.inp_template = self.template_store.get_template(self.init_prop) if self.template_store is not None else None
        self.root_spec = None
        self.proof_tree = None
        self.root_lb = None
        self.root_ub = None

        self.cur_specs = self.get_init_specs(init_prop)
        self.tree_size = len(self.cur_specs)
        self.prev_lb = None
        self.cur_lb = None
        self.nodes_visited = 0
        self.pruned_nodes = 0
        self.adv_example_nodes = 0
        self.max_depth_reached = 0
        self.depth_histogram = {}
        self.first_k_chosen_splits = []
        self.first_k_node_lbs = []
        self.trace_event_index = 0
        self.iteration_index = 0
        self.frontier_snapshots = []
        self.logged_split_effectiveness = set()

    def get_init_specs(self, init_prop):
        tree_avail = self.template_store is not None and self.template_store.is_tree_available(init_prop)

        if tree_avail and type(self.args.pt_method) == IVAN:
            proof_tree = self.template_store.get_proof_tree(init_prop)
            cur_specs = proof_tree.get_pruned_leaves(self.args.pt_method.threshold, self.split)
        elif tree_avail and self.args.pt_method == ProofTransferMethod.REUSE:
            proof_tree = self.template_store.get_proof_tree(init_prop)
            cur_specs = proof_tree.get_leaves()
        else:
            unstable_relus = self.get_unstable_relus()
            cur_specs = self.create_initial_specs(init_prop, unstable_relus)
        return cur_specs

    def get_unstable_relus(self):
        lb, is_feasible, adv_ex = self.transformer.compute_lb(complete=True)
        self.root_lb = lb
        self.root_ub = self.get_output_upper_bound()
        status, _ = self.get_status_details(adv_ex, is_feasible, lb)

        if 'unstable_relus' in dir(self.transformer):
            unstable_relus = self.transformer.unstable_relus
        else:
            unstable_relus = None

        if status != Status.UNKNOWN:
            self.global_status = status
            if status == Status.VERIFIED and self.print_result:
                print(status)
        return unstable_relus

    def run(self):
        """
        It is the public method called from the analyzer. @param split is a string that chooses the mode for relu
        or input splitting.
        """
        split_score = self.set_split_score(self.init_prop, self.cur_specs, inp_template=self.inp_template)
        self.log_root_artifacts(split_score)

        if self.global_status != Status.UNKNOWN:
            self.log_search_summary()
            return

        while self.continue_search():
            self.iteration_index += 1
            self.update_depth()
            self.record_frontier_snapshot(stage="pre_verify", frontier=self.cur_specs)

            self.prev_lb = self.cur_lb
            self.reset_cur_lb()

            # Main verification loop
            if self.args.parallel:
                self.verify_specs_parallel()
            else:
                self.verify_specs()

            self.record_split_effectiveness()
            self.record_frontier_snapshot(stage="post_verify", frontier=self.cur_specs)

            # Each spec should hold the prev lb and current lb
            self.cur_specs, verified_specs = branch.branch_unsolved(self.cur_specs, self.split, split_score=split_score,
                                                                    inp_template=self.inp_template, args=self.args,
                                                                    net=self.net, transformer=self.transformer,
                                                                    branch_observer=self.record_split_event,
                                                                    candidate_limit=self.get_candidate_limit())
            # Update the tree size
            self.tree_size += len(self.cur_specs)
            self.record_frontier_snapshot(stage="post_branch", frontier=self.cur_specs)

        self.check_verified_status()
        self.store_final_tree()
        self.log_search_summary()

    def verify_specs(self):
        for spec in self.cur_specs:
            self.update_transformer(spec.input_spec, relu_spec=spec.relu_spec)

            # Transformer is updated with new mask
            status, lb, details = self.verify_node(self.transformer, spec.input_spec)
            self.update_cur_lb(lb)
            spec.update_status(status, lb)
            self.record_node_result(spec, status, lb, details)

            if status == Status.ADV_EXAMPLE or self.is_timeout():
                self.global_status = status
                self.store_final_tree()
                self.log_search_summary()
                return

    def verify_specs_parallel(self):
        cur_specs = [self.cur_specs[i*self.args.batch_size:(i+1)*self.args.batch_size] for i in range((len(self.cur_specs)+self.args.batch_size-1)//self.args.batch_size)]

        for batch_specs in cur_specs:
            # Create a batch specification
            batch_input_spec = merge_input_specs([spec.input_spec for spec in batch_specs])
            self.transformer = get_domain_transformer(self.args, self.net, batch_input_spec, complete=True)
            lb, is_feasible, adv_ex = self.transformer.compute_lb(complete=True)

            for i in range(len(batch_specs)):
                status, reason = self.get_status_details(adv_ex, is_feasible, lb[i])
                self.update_cur_lb(lb[i])
                batch_specs[i].update_status(status, lb[i])
                self.record_node_result(
                    batch_specs[i],
                    status,
                    lb[i],
                    {"is_feasible": bool(is_feasible), "has_adversarial_example": adv_ex is not None, "reason": reason},
                )

                if status == Status.ADV_EXAMPLE or self.is_timeout():
                    self.global_status = status
                    self.store_final_tree()
                    self.log_search_summary()
                    return

    def store_final_tree(self):
        if self.template_store is None:
            return
        self.proof_tree = ProofTree(self.root_spec)
        self.template_store.add_tree(self.init_prop, self.proof_tree)

    def verify_node(self, transformer, prop):
        """
        It is called from bnb_relu_complete. Attempts to verify (ilb, iub), there are three possible outcomes that
        are indicated by the status: 1) verified 2) adversarial example is found 3) Unknown
        """
        lb, is_feasible, adv_ex = transformer.compute_lb(complete=True)

        status, reason = self.get_status_details(adv_ex, is_feasible, lb)
        return status, lb, {"is_feasible": bool(is_feasible), "has_adversarial_example": adv_ex is not None,
                            "reason": reason}

    def get_status(self, adv_ex, is_feasible, lb):
        status, _ = self.get_status_details(adv_ex, is_feasible, lb)
        return status

    def get_status_details(self, adv_ex, is_feasible, lb):
        status = Status.UNKNOWN
        reason = "unknown"
        if adv_ex is not None:
            config.write_log("Found a counter example!")
            status = Status.ADV_EXAMPLE
            reason = "adversarial_example"
        elif (not is_feasible) or (lb is not None and torch.all(lb >= 0)):
            status = Status.VERIFIED
            reason = "infeasible_pruned" if not is_feasible else "verified_lb_nonnegative"
        return status, reason

    def update_transformer(self, prop, relu_spec=None):
        relu_mask = None
        if relu_spec is not None:
            relu_mask = relu_spec.relu_mask

        if 'update_spec' in dir(self.transformer) and is_relu_split(self.args.split):
            self.transformer.update_spec(prop, relu_mask=relu_mask)
        else:
            self.transformer = get_domain_transformer(self.args, self.net, prop, complete=True)

    def check_verified_status(self):
        # Verified
        if len(self.cur_specs) == 0:
            self.global_status = Status.VERIFIED
            if self.print_result:
                print(Status.VERIFIED)

    def reset_cur_lb(self):
        self.cur_lb = None

    def is_timeout(self):
        cur_time = (time.time() - self.init_time)
        ret = self.args.timeout is not None and cur_time > self.args.timeout
        return ret

    def continue_search(self):
        return self.global_status == Status.UNKNOWN and len(self.cur_specs) > 0 and (not self.is_timeout())

    def update_cur_lb(self, lb):
        # lb can be None if the LP is infeasible
        if lb is not None:
            if self.cur_lb is None:
                self.cur_lb = lb
            else:
                self.cur_lb = min(lb, self.cur_lb)

    def update_depth(self):
        #print('Depth :', self.depth, 'Specs size :', len(self.cur_specs), 'LB:', self.cur_lb)
        self.depth += 1

    def create_initial_specs(self, prop, unstable_relus):
        if is_relu_split(self.split):
            relu_spec = specs.create_relu_spec(unstable_relus)
            self.root_spec = specs.Spec(prop, relu_spec=relu_spec, status=self.global_status)
            cur_specs = specs.SpecList([self.root_spec])
            config.write_log("Unstable relus: " + str(unstable_relus))
        else:
            if self.args.initial_split:
                # Do a smarter initial split similar to ERAN
                # This works only for ACAS-XU
                zono_transformer = ZonoTransformer(prop, complete=True)
                zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)

                center = zono_transformer.centers[-1]
                cof = zono_transformer.cofs[-1]
                cof_abs = torch.sum(torch.abs(cof), dim=0)
                lb = center - cof_abs
                adv_index = torch.argmin(lb)
                input_len = len(prop.input_lb)
                smears = torch.abs(cof[:input_len, adv_index])
                split_multiple = 10 / torch.sum(smears)  # Dividing the initial splits in the proportion of above score
                num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]

                inp_specs = prop.multiple_splits(num_splits)
                cur_specs = specs.SpecList([specs.Spec(prop, status=self.global_status) for prop in inp_specs])
                # TODO: Add a root spec in this case as well
            else:
                self.root_spec = specs.Spec(prop, status=self.global_status)
                cur_specs = specs.SpecList([self.root_spec])

        return cur_specs

    def set_split_score(self, prop, relu_mask_list, inp_template=None):
        """
        Computes relu score for each relu if the split method needs it. Otherwise, returns None
        """
        split_score = None
        if self.split == Split.RELU_GRAD:
            # These scores only work for torch models
            split_score = specs.score_relu_grad(relu_mask_list[0], prop, net=self.net)
        elif self.split == Split.RELU_ESIP_SCORE or self.split == Split.RELU_KFSB:
            # These scores only work with deepz transformer
            zono_transformer = ZonoTransformer(prop, complete=True)
            zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)
            split_score = specs.score_relu_esip(zono_transformer)
        elif self.split == Split.RELU_ESIP_SCORE2:
            # These scores only work with deepz transformer
            zono_transformer = ZonoTransformer(prop, complete=True)
            zono_transformer = nnverify.domains.build_transformer(zono_transformer, self.net, prop)
            split_score = specs.score_relu_esip2(zono_transformer)

        # Update the scores based on previous scores
        if inp_template is not None and split_score is not None and self.template_store is not None:
            if type(self.args.pt_method) == IVAN or type(self.args.pt_method) == REORDERING:
                # compute mean worst case improvements
                observed_split_scores = self.template_store.get_proof_tree(prop).get_observed_split_score()
                alpha = self.args.pt_method.alpha
                thr = self.args.pt_method.threshold

                for chosen_split in observed_split_scores:
                    if chosen_split in split_score and observed_split_scores[chosen_split] < self.args.pt_method.threshold:
                        split_score[chosen_split] = alpha * split_score[chosen_split] + (1 - alpha) * (
                                observed_split_scores[chosen_split] - thr)

        return split_score

    def get_candidate_limit(self):
        if self.artifact_logger is not None:
            return self.artifact_logger.top_k_candidates
        return 5

    def get_output_upper_bound(self):
        try:
            if hasattr(self.transformer, "compute_ub"):
                ub = self.transformer.compute_ub()
                if isinstance(ub, tuple):
                    return ub[0]
                return ub
            if hasattr(self.transformer, "get_all_bounds"):
                _, upper_bounds = self.transformer.get_all_bounds()
                if len(upper_bounds) > 0:
                    return upper_bounds[-1]
        except Exception:
            return None
        return None

    def log_root_artifacts(self, split_score):
        if self.artifact_logger is None:
            return

        unstable_relus = getattr(self.transformer, "unstable_relus", None)
        root_candidates = []
        if self.root_spec is not None:
            root_candidates = branch.get_top_candidates(self.root_spec, self.split, split_score=split_score,
                                                        limit=self.get_candidate_limit())

        self.artifact_logger.set_root_artifacts(
            transformer=self.transformer,
            unstable_relus=unstable_relus,
            root_lower_bound=self.root_lb,
            root_upper_bound=self.root_ub,
            root_candidates=root_candidates,
        )

    def record_node_result(self, spec, status, lb, details):
        depth = branch.get_spec_depth(spec)
        self.nodes_visited += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
        self.depth_histogram[depth] = self.depth_histogram.get(depth, 0) + 1

        if len(self.first_k_node_lbs) < self.get_summary_prefix():
            self.first_k_node_lbs.append(branch._scalar_lb(lb))

        if status == Status.VERIFIED:
            self.pruned_nodes += 1
            self.record_trace_event(
                {
                    "event_type": "pruned",
                    "node_depth": depth,
                    "node_lower_bound": branch._scalar_lb(lb),
                    "node_signature": branch.summarize_node_state(spec),
                    "status": status.name,
                    "prune_reason": details.get("reason"),
                }
            )
        elif status == Status.ADV_EXAMPLE:
            self.adv_example_nodes += 1
            self.record_trace_event(
                {
                    "event_type": "adv_example",
                    "node_depth": depth,
                    "node_lower_bound": branch._scalar_lb(lb),
                    "node_signature": branch.summarize_node_state(spec),
                    "status": status.name,
                    "prune_reason": details.get("reason"),
                }
            )

    def record_split_event(self, event):
        chosen_split = event.get("chosen_split")
        if len(self.first_k_chosen_splits) < self.get_summary_prefix():
            self.first_k_chosen_splits.append(chosen_split)
        self.record_trace_event({"event_type": "split", "iteration_index": self.iteration_index, **event})

    def record_trace_event(self, event):
        if self.artifact_logger is None:
            return
        self.trace_event_index += 1
        self.artifact_logger.append_search_event({"event_index": self.trace_event_index, **event})

    def get_summary_prefix(self):
        if self.artifact_logger is not None:
            return self.artifact_logger.summary_prefix
        return 20

    def log_search_summary(self):
        if self.artifact_logger is None:
            return
        self.artifact_logger.set_search_summary(self.get_search_summary())

    def get_search_summary(self):
        return {
            "total_nodes_visited": self.nodes_visited,
            "total_pruned_nodes": self.pruned_nodes,
            "total_adversarial_nodes": self.adv_example_nodes,
            "maximum_depth_reached": self.max_depth_reached,
            "depth_histogram": self.depth_histogram,
            "first_k_chosen_splits": self.first_k_chosen_splits,
            "first_k_node_lower_bounds": self.first_k_node_lbs,
            "frontier_size_at_exit": len(self.cur_specs),
            "leaf_count": self.count_leaves(),
            "global_status": self.global_status.name,
            "frontier_snapshots": self.frontier_snapshots,
        }

    def count_leaves(self):
        if self.root_spec is None:
            return len(self.cur_specs)

        leaves = 0
        queue = [self.root_spec]
        while len(queue) != 0:
            node = queue.pop()
            if len(node.children) == 0:
                leaves += 1
            else:
                queue.extend(node.children)
        return leaves

    def record_frontier_snapshot(self, stage, frontier):
        if len(self.frontier_snapshots) >= self.get_summary_prefix():
            return

        node_lbs = [branch._scalar_lb(spec.lb) for spec in frontier if spec.lb is not None]
        scalar_node_lbs = [lb for lb in node_lbs if isinstance(lb, (int, float))]
        status_counts = {}
        for spec in frontier:
            status_counts[spec.status.name] = status_counts.get(spec.status.name, 0) + 1

        snapshot = {
            "iteration_index": self.iteration_index,
            "stage": stage,
            "depth_counter": self.depth,
            "frontier_size": len(frontier),
            "status_counts": status_counts,
            "current_global_lower_bound": branch._scalar_lb(self.cur_lb),
            "previous_global_lower_bound": branch._scalar_lb(self.prev_lb),
            "frontier_min_node_lb": min(scalar_node_lbs) if len(scalar_node_lbs) > 0 else None,
            "frontier_max_node_lb": max(scalar_node_lbs) if len(scalar_node_lbs) > 0 else None,
        }
        self.frontier_snapshots.append(snapshot)

    def record_split_effectiveness(self):
        parent_nodes = []
        seen = set()
        for spec in self.cur_specs:
            parent = spec.parent
            if parent is None:
                continue
            parent_id = id(parent)
            if parent_id in seen or parent_id in self.logged_split_effectiveness:
                continue
            seen.add(parent_id)
            parent_nodes.append(parent)

        for parent in parent_nodes:
            if len(parent.children) == 0:
                continue
            if any(child.status == Status.UNKNOWN for child in parent.children):
                continue

            parent_lb = branch._scalar_lb(parent.lb)
            child_lbs = [branch._scalar_lb(child.lb) for child in parent.children]
            scalar_child_lbs = [lb for lb in child_lbs if isinstance(lb, (int, float))]
            min_child_lb = min(scalar_child_lbs) if len(scalar_child_lbs) > 0 else None
            max_child_lb = max(scalar_child_lbs) if len(scalar_child_lbs) > 0 else None
            mean_child_lb = sum(scalar_child_lbs) / len(scalar_child_lbs) if len(scalar_child_lbs) > 0 else None

            if isinstance(parent_lb, (int, float)) and min_child_lb is not None:
                min_improvement = min_child_lb - parent_lb
                max_improvement = max_child_lb - parent_lb
                mean_improvement = mean_child_lb - parent_lb
            else:
                min_improvement = None
                max_improvement = None
                mean_improvement = None

            self.logged_split_effectiveness.add(id(parent))
            self.record_trace_event(
                {
                    "event_type": "split_effectiveness",
                    "iteration_index": self.iteration_index,
                    "parent_depth": branch.get_spec_depth(parent),
                    "parent_signature": branch.summarize_node_state(parent),
                    "chosen_split": branch.serialize_split_identifier(parent.chosen_split),
                    "parent_lower_bound": parent_lb,
                    "child_lower_bounds": child_lbs,
                    "child_statuses": [child.status.name for child in parent.children],
                    "min_improvement": min_improvement,
                    "mean_improvement": mean_improvement,
                    "max_improvement": max_improvement,
                }
            )
