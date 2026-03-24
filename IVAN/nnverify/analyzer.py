import torch

import nnverify.attack
import nnverify.domains
import nnverify.util as util
import nnverify.specs.spec as specs
import time
import nnverify.bnb.bnb as bnb

from nnverify import config
from nnverify.common import Status
from nnverify.specs.out_spec import OutSpecType
from nnverify.common.result import Result, Results
from nnverify.proof_transfer.template import TemplateStore
from nnverify.domains import build_transformer, get_domain_transformer


class Analyzer:
    def __init__(self, args, net=None, template_store=None, artifact_logger=None, enable_template_store=True):
        """
        @param args: configuration arguments for the analyzer such as the network, domain, dataset, attack, count, dataset,
            epsilon and split
        """
        self.args = args
        self.net = net
        self.template_store = template_store
        self.artifact_logger = artifact_logger
        self.enable_template_store = enable_template_store
        self.timeout = args.timeout
        self.device = config.DEVICE
        self.transformer = None
        self.init_time = None

        if self.net is None:
            self.net = util.get_net(self.args.net, self.args.dataset)
        if self.template_store is None and self.enable_template_store:
            self.template_store = TemplateStore()

    def analyze(self, prop):
        self.update_transformer(prop)
        tree_size = 1
        leaf_count = 1
        search_summary = None

        if self.artifact_logger is not None:
            self.log_center_input_behavior(prop)

        # Check if classified correctly
        if nnverify.attack.check_adversarial(prop.input, self.net, prop):
            return Status.MISS_CLASSIFIED, tree_size, leaf_count, search_summary

        # Check Adv Example with an Attack
        if self.args.attack is not None:
            adv = self.args.attack.search_adversarial(self.net, prop, self.args)
            if nnverify.attack.check_adversarial(adv, self.net, prop):
                return Status.ADV_EXAMPLE, tree_size, leaf_count, search_summary

        if self.args.split is None:
            status = self.analyze_no_split()
        elif self.args.split is None:
            status = self.analyze_no_split_adv_ex(prop)
        else:
            bnb_analyzer = bnb.BnB(
                self.net,
                self.transformer,
                prop,
                self.args,
                self.template_store,
                artifact_logger=self.artifact_logger,
            )
            if self.args.parallel:
                bnb_analyzer.run_parallel()
            else:
                bnb_analyzer.run()

            status = bnb_analyzer.global_status
            tree_size = bnb_analyzer.tree_size
            leaf_count = bnb_analyzer.count_leaves()
            search_summary = bnb_analyzer.get_search_summary()
        return status, tree_size, leaf_count, search_summary

    def log_center_input_behavior(self, prop):
        center_input = prop.input if prop.input is not None else (prop.input_lb + prop.input_ub) / 2
        if center_input is None:
            return

        center_input = center_input.flatten().detach().cpu()
        adv_label, output = util.compute_output_tensor(center_input, self.net)
        output = output.detach().cpu()
        true_label = prop.get_label().item() if prop.is_local_robustness() else None
        sorted_output, _ = torch.sort(output.flatten(), descending=True)
        top2_margin = float(sorted_output[0].item() - sorted_output[1].item()) if sorted_output.numel() > 1 else None
        true_label_margin = None
        if true_label is not None and output.numel() > 1:
            competing_logits = torch.cat([output[:true_label], output[true_label + 1:]])
            if competing_logits.numel() > 0:
                true_label_margin = float((output[true_label] - torch.max(competing_logits)).item())

        constraint_margin = None
        constraint_margin_summary = None
        if prop.out_constr.constr_mat is not None:
            constr_weight, constr_bias = prop.out_constr.constr_mat
            constraint_margin = output @ constr_weight + constr_bias
            constraint_margin_summary = {
                "min": float(torch.min(constraint_margin).item()),
                "max": float(torch.max(constraint_margin).item()),
                "mean": float(torch.mean(constraint_margin.float()).item()),
            }

        self.artifact_logger.set_center_input_behavior(
            center_input=center_input,
            output=output,
            predicted_label=adv_label,
            true_label=true_label,
            constraint_margin=constraint_margin,
            constraint_margin_summary=constraint_margin_summary,
            center_source="stored_input" if prop.input is not None else "box_center",
            top2_margin=top2_margin,
            true_label_margin=true_label_margin,
        )

    def update_transformer(self, prop):
        if self.transformer is not None and 'update_input' in dir(self.transformer) \
                and prop.out_constr.constr_type == OutSpecType.LOCAL_ROBUST:
            self.transformer.update_input(prop)
        else:
            self.transformer = get_domain_transformer(self.args, self.net, prop, complete=True)

    def analyze_no_split_adv_ex(self, prop):
        # TODO: handle feasibility
        lb, _, adv_ex = self.transformer.compute_lb()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        elif adv_ex is not None:
            status = Status.ADV_EXAMPLE
        print(lb)
        return status

    def analyze_no_split(self):
        lb = self.transformer.compute_lb()
        status = Status.UNKNOWN
        if torch.all(lb >= 0):
            status = Status.VERIFIED
        print('LB: ', lb)
        return status

    def run_analyzer(self, props=None):
        """
        Prints the output of verification - count of verified, unverified and the cases for which the adversarial example
            was found
        """
        print('Running on the network: ', self.args.net)
        print('Number of verification instances: ', self.args.count)
        print('Timeout of verification: ', self.args.timeout)
        print('Using %s abstract domain' % self.args.domain)

        if props is None:
            props, inputs = specs.get_specs(self.args.dataset, spec_type=self.args.spec_type, count=self.args.count,
                                            eps=self.args.eps)

        results = self.analyze_domain(props)

        results.compute_stats()
        print('Results: ', results.output_count)
        print('Average time:', results.avg_time)
        return results

    # There are multiple clauses in the inout specification
    # Property should hold on all the input clauses
    @staticmethod
    def extract_status(cl_status):
        for status in cl_status:
            if status != Status.VERIFIED:
                return status
        return Status.VERIFIED

    def analyze_domain(self, props):
        results = Results(self.args)
        for i in range(len(props)):
            print("************************** Proof %d *****************************" % (i+1))
            num_clauses = props[i].get_input_clause_count()
            clause_ver_status = []
            ver_start_time = time.time()

            for j in range(num_clauses):
                clause_prop = props[i].get_input_clause(j)
                if self.artifact_logger is not None:
                    self.artifact_logger.start_property(clause_prop, property_index=i, clause_index=j, args=self.args)

                clause_start_time = time.time()
                cl_status, tree_size, leaf_count, search_summary = self.analyze(clause_prop)
                clause_ver_status.append(cl_status)
                clause_runtime = time.time() - clause_start_time

                if self.artifact_logger is not None:
                    self.artifact_logger.finalize_property(
                        cl_status,
                        runtime_sec=clause_runtime,
                        tree_size=tree_size,
                        leaf_count=leaf_count,
                    )

            status = self.extract_status(clause_ver_status)
            print(status)
            ver_time = time.time() - ver_start_time
            results.add_result(Result(ver_time, status, tree_size=tree_size))

        return results
