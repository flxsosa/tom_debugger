import itertools
from copy import deepcopy, copy

import numpy as np

import ElementExtractor
import utils
import probs

class BayesianInferenceModel:
    def __init__(
        self,
        variables,
        context,
        llm,
        verbose,
        inf_agent,
        model_name,
        dataset_name,
        episode_name,
        answer_choices,
        K,
        all_prob_estimations=None,
        no_observation_hypothesis="NONE",
        reduce_hypotheses=False,
    ):
        if all_prob_estimations is None:
            all_prob_estimations = {}
        self.model_name = model_name
        self.episode_name = episode_name
        self.K = K
        self.dataset_name = dataset_name
        self.answer_choices = answer_choices
        new_variables = []
        for i, var in enumerate(variables):
            if isinstance(var, list):
                if isinstance(var[0], float):
                    continue
            if verbose:
                print(var)
            if "'s " in var.name:
                orig_var_name = deepcopy(var.name)
                variables[i].name = var.name.split("'s ")[1]
            new_variables.append(variables[i])
        variables = new_variables
        self.previous_observation_hypotheses = set()

        self.variables = {var.name: var for var in variables}
        self.context = context
        self.llm = llm
        self.inf_agent = inf_agent
        self.parent_graph = {
            "Observation": ["State"],
            "Belief": ["Previous Belief", "Observation"],
            "Action": ["Goal", "Belief", "Emotion", "Belief of Goal"],
            "Utterance": ["Goal", "Belief", "Emotion"],
            "Emotion": ["Goal", "Belief"],
            "Expression": ["Emotion"],
            "Response": ["Goal", "Belief", "Utterance"],
        }
        self.recorder = all_prob_estimations
        self.verbose = verbose
        self.no_observation_hypothesis = no_observation_hypothesis
        self.reduce_hypotheses = reduce_hypotheses

        # Check for "Observation" first; if not found, use "State" as the observation
        observation_var_name = next(
            (name for name in self.variables if "Observation" in name),
            next(name for name in self.variables if "State" in name),
        )

        initial_observations = self.variables[observation_var_name].possible_values
        self.previous_observation_hypotheses.update(initial_observations)

    def rewrite_graph(self):
        """Remove variables not in self.variables from parent_graph while
        preserving dependencies.

        Performs topological sorting with graph contraction to eliminate
        missing variables from the Bayesian network structure. When a variable
        is removed, its dependencies are transitively connected to its children
        to maintain proper inference relationships.

        Args:
            None (uses self.parent_graph and self.variables)

        Returns:
            None (modifies self.parent_graph in place)
        """
        # Copy original graph
        new_parent_graph = deepcopy(self.parent_graph)
        # Track incoming edges count
        in_degree = {}
        # Parent → Children mapping  
        son_graph = {}
        # Processing queue
        stack = []

        for key, val in new_parent_graph.items():
            # Count parents for each variable
            in_degree[key] = len(val)
            for par in val:
                if par in son_graph:
                    # Add child to parent's list
                    son_graph[par].append(key)
                else:
                    # Create new parent entry
                    son_graph[par] = [key]

        all_variables = set()
        # Add child variables
        for key, val in new_parent_graph.items():
            all_variables.add(key)
            # Add parent variables
            for v in val:
                all_variables.add(v)

        for key in all_variables:
            if key not in in_degree or in_degree[key] == 0:
                # Variables with no parents go first
                stack.append(key)
        # Process variables in topological order
        while len(stack):
            now_var = stack[0]
            if now_var not in self.variables:
                # Parent of now_var
                left = new_parent_graph[now_var] if now_var in new_parent_graph else []
                # Children of now_var
                right = son_graph[now_var] if now_var in son_graph else []
                for j in right:
                    new_parent_graph[j].remove(now_var)
                    for i in left:
                        if i not in new_parent_graph[j]:
                            new_parent_graph[j].append(i)
                l = list(new_parent_graph.keys())
                for key in l:
                    if key == now_var:
                        del new_parent_graph[now_var]
                        continue
                    for v in new_parent_graph[key]:
                        if v == now_var:
                            new_parent_graph[key] = val.remove(v)
                            break
            stack.pop(0)
            if now_var not in son_graph:
                continue
            for son in son_graph[now_var]:
                in_degree[son] -= 1
                if in_degree[son] == 0:
                    stack.append(son)  # New root variable
        self.parent_graph = new_parent_graph
        if self.verbose:
            utils.enh_print(f"New graph: {new_parent_graph}", color="red")

    def recompute_combinations(self, left, infer_var_name):
        """Generate all possible combinations of latent variable values.

        Args:
            left: List of latent variable names
            infer_var_name: Name of variable being inferred

        Returns:
            List of all possible combinations of latent variable values
        """
        combo = []
        for unob_var_name in left:
            if unob_var_name == infer_var_name:
                continue
            no_prior = True
            if isinstance(
                self.variables[unob_var_name].prior_probs, np.ndarray):
                no_prior = False
            tmp = []
            for i, val in enumerate(self.variables[unob_var_name].possible_values):
                if no_prior:
                    tmp.append((val, 1.0))
                else:
                    tmp.append((val, self.variables[unob_var_name].prior_probs[i]))
            combo.append(tmp)
        return list(itertools.product(*combo))

    def calculate_prob_product(self, var_dict, calc):
        """Calculate the probability product of a variable's possible values.

        Args:
            var_dict: Dictionary of variable names and their possible values
            calc: List of tuples (variable name, list of parent variables)

        Returns:
            Tuple containing:
            - Probability product
            - Dictionary of individual likelihoods
            - List of tuples (variable name, list of parent variables, variable value, likelihood)
        """
        prob = 1.0
        individual_likelihoods = {}
        node_results_tracker = []

        for son, parents in calc:
            info_var = []
            for parent in parents:
                if "State" in parent:
                    if "Previous" in parent:
                        if (
                            "no new observations compared to previous timestamp" not in var_dict[son]
                        ):
                            continue
                    info_var.append(f"{parent}: {var_dict[parent]}")
                else:
                    info_var.append(f"{self.inf_agent}'s {parent}: {var_dict[parent]}")

            info = "\n".join(info_var)
            key = info + ";" + var_dict[son]

            # Action and utterance key might be the same.
            if son == "Utterance":
                key = "Utterance:" + key

            if self.no_observation_hypothesis in key and son == "Belief":
                # in this case, belief == previous belief -> 1.0; belief != previous belief -> 0.03
                if "Previous Belief" in var_dict:
                    if var_dict["Previous Belief"] == var_dict["Belief"]:
                        logits = 1.0
                    else:
                        logits = 0.03
            # No utterance, do not calculate
            elif "Utterance" in son and var_dict[son] == "NONE":
                logits = 0.03
            else:
                if "Action" in son and (
                    self.dataset_name == "BigToM_fatb"
                    or self.dataset_name == "BigToM_fafb"
                ):
                    info = [info, (self.answer_choices)]
                    logits = probs.get_likelihood(
                        info,
                        f"{var_dict[son]}",
                        model=self.llm,
                        verbose=self.verbose,
                        variable="Actions",
                        inf_agent=self.inf_agent,
                    )
                if key in self.recorder:
                    logits = self.recorder[key]
                else:
                    logits = probs.get_likelihood(
                        info,
                        f"{var_dict[son]}",
                        model=self.llm,
                        verbose=self.verbose,
                        variable=son,
                        inf_agent=self.inf_agent,
                    )
                    # print(logits)

            self.recorder[key] = logits

            node_results_tracker.append((son, parents, copy(var_dict), logits))
            individual_likelihoods[(son, tuple(parents), var_dict[son])] = logits

            prob *= logits

        return prob, individual_likelihoods, node_results_tracker

    def reduce_obs_hypospace(self):
        all_node_results = []
        obs_hypos = self.variables["Observation"].possible_values
        state_value = self.variables["State"].possible_values[0]
        var_dict = {}
        var_dict["State"] = state_value
        po = []
        for o in obs_hypos:
            var_dict["Observation"] = o
            p, ind_lh, node_result = self.calculate_prob_product(
                var_dict, [("Observation", ["State"])]
            )
            all_node_results.extend(node_result)
            po.append(p)
        new_hypos = []
        for i, o in enumerate(obs_hypos):
            # For a possible observation hypotheses,
            # if the likelihood P(o_i|s) is lower than 0.03
            # we discard it
            if po[i] <= 0.03:
                continue
            new_hypos.append(o)
        if len(new_hypos) == 0:
            new_hypos = obs_hypos
        self.variables["Observation"].possible_values = new_hypos
        return all_node_results

    def reduce_belief_hypospace(self):
        all_node_results = []
        belief_hypos = self.variables["Belief"].possible_values

        if len(belief_hypos) <= 1:
            return all_node_results
        if "Action" not in self.variables:
            return all_node_results

        if "Goal" in self.variables:
            if len(self.variables["Goal"].possible_values) > 1:
                # goal is also not determined, then return
                return all_node_results
            else:
                # goal is determined, reduce belief hypotheses
                with_goal = True
        else:
            # goal not included in the model, reduce belief hypotheses
            with_goal = False

        action_value = self.variables["Action"].possible_values[0]
        if "Utterance" in self.variables:
            utterance_value = self.variables["Utterance"].possible_values[0]
        else:
            utterance_value = (
                "NONE"  # Utterance not in the model, likelihood will be minimum
            )
        var_dict = {}
        var_dict["Action"] = action_value
        var_dict["Utterance"] = utterance_value
        pa, pu = [], []
        for b in belief_hypos:
            var_dict["Belief"] = b
            if with_goal:
                parents = ["Belief", "Goal"]
                var_dict["Goal"] = self.variables["Goal"].possible_values[0]
            else:
                parents = ["Belief"]
            p, ind_lh, node_result = self.calculate_prob_product(
                var_dict, [("Action", parents)]
            )
            all_node_results.extend(node_result)
            pa.append(p)
            p, ind_lh, node_result = self.calculate_prob_product(
                var_dict, [("Utterance", parents)]
            )
            all_node_results.extend(node_result)
            pu.append(p)
        new_hypos = []
        for i, b in enumerate(belief_hypos):
            if pa[i] <= 0.03 and pu[i] <= 0.03:
                continue
            new_hypos.append(b)
        if len(new_hypos) == 0:
            new_hypos = belief_hypos
        self.variables["Belief"].possible_values = new_hypos
        return all_node_results

    def infer(self, infer_var_name, model_name, episode_name, init_belief=False):
        """
        Infer the variable of interest using Bayesian Inference.

        Args:
            infer_var_name: The variable to infer.
            model_name: The name of the model.
            episode_name: The name of the episode.
            init_belief: Whether to initialize the belief.

        Returns:
            The probabilities of the variable of interest.
            The recorder.
            The node results.
        """
        if (
            "BigToM" in self.dataset_name and
            init_belief and
            "Belief" in self.variables
        ):
            # NOTE: init_belief means that there are no actions of the agent
            # so we only calculate P(belief0) and then use it for other
            # calculations if needed
            # TODO: make init_belief dynamic depending on how many timestamps
            if infer_var_name == "Action":
                # calculate P(initial Belief)
                initial_belief_vals = self.variables["Belief"].possible_values
                probs = []
                for b in initial_belief_vals:
                    logits = probs.get_likelihood(
                        self.context,
                        b,
                        model=self.llm,
                        verbose=self.verbose,
                        variable="Initial Belief",
                        inf_agent=self.inf_agent,
                    )
                    probs.append(logits)
                exps = np.exp(probs)
                probs = exps / np.sum(exps)
                max_prob = max(probs)
                max_prob = np.where(probs == max_prob)[0][0]
                init_belief = initial_belief_vals[max_prob]
                # init belief is not uniform
                new_belief = ElementExtractor.Variable(
                    name="Belief",
                    in_model=True,
                    is_observed=True,
                    possible_values=[init_belief],
                    prior_probs=max(probs),
                )
                new_prev_belief = ElementExtractor.Variable(
                    name="Previous Belief",
                    in_model=True,
                    is_observed=True,
                    possible_values=[init_belief],
                    prior_probs=max(probs),
                )

                # NOTE: since there are no actions from the main_agent then we
                # just calculate P(init belief) and use it for action
                # likelihood estimation
                self.variables["Belief"] = (
                    new_belief
                )
                self.variables["Previous Belief"] = new_prev_belief

            elif infer_var_name == "Belief":
                initial_belief_vals = self.variables["Belief"].possible_values
                probs = []

                for b in initial_belief_vals:
                    logits = probs.get_likelihood(
                        self.context,
                        b,
                        model=self.llm,
                        verbose=self.verbose,
                        variable="Initial Belief",
                        inf_agent=self.inf_agent,
                    )
                    probs.append(logits)
                exps = np.exp(probs)
                probs = exps / np.sum(exps)
                if len(probs) == 1:
                    return (
                        probs,
                        self.recorder,
                        [
                            (
                                "Initial Belief",
                                ["Story"],
                                {
                                    "Story": self.context,
                                    "Initial Belief": initial_belief_vals[0],
                                },
                                probs[0],
                            ),
                        ],
                    )
                return (
                    probs,
                    self.recorder,
                    [
                        (
                            "Initial Belief",
                            ["Story"],
                            {
                                "Story": self.context,
                                "Initial Belief": initial_belief_vals[0],
                            },
                            probs[0],
                        ),
                        (
                            "Initial Belief",
                            ["Story"],
                            {
                                "Story": self.context,
                                "Initial Belief": initial_belief_vals[1],
                            },
                            probs[1],
                        ),
                    ],
                )
        # Rewrite graph to remove variables not in self.variables
        self.rewrite_graph()

        left = []
        right = []
        # Add observed variables to right, latent variables to left
        for key, var in self.variables.items():
            if var.is_observed:
                right.append(var.name)
            else:
                left.append(var.name)
        if self.verbose:
            print(f'Estimating P({",".join(left)}|{",".join(right)})', end="")

        all_node_results = []
        if (
            "Belief" in self.variables and
            len(self.variables["Belief"].possible_values) == 1
        ):
            print("Not considering Observation in the model")
            if "Observation" in self.parent_graph:
                del self.parent_graph["Observation"]
            if "Belief" in self.parent_graph:
                del self.parent_graph["Belief"]
        elif self.reduce_hypotheses:
            if "Observation" in left:
                all_node_results += self.reduce_obs_hypospace()
            if "Belief" in left and infer_var_name != "Belief":
                all_node_results += self.reduce_belief_hypospace()
        # Generate all possible combinations of latent variable values
        combo = self.recompute_combinations(left, infer_var_name)

        var_dict = {}
        # Set observed variables to their possible values
        for ob_var_name in right:
            var_dict[ob_var_name] = self.variables[ob_var_name].possible_values[0]

        all_var_names = left + right
        # List of tuples (variable name, list of parent variables)
        calc = []
        if self.verbose:
            print("=", end="")
        for var_name in all_var_names:
            if var_name in self.parent_graph:
                need_compute = False
                if var_name in left:
                    need_compute = True
                for par_var_name in self.parent_graph[var_name]:
                    if par_var_name in left:
                        need_compute = True
                if not need_compute:
                    continue
                calc.append((var_name, self.parent_graph[var_name]))
                if self.verbose:
                    print(
                        f'P({var_name}|{",".join(self.parent_graph[var_name])})', end=""
                    )

        try:
            infer_var = self.variables[infer_var_name]
        except KeyError:
            print(f"No {infer_var_name}")
            return

        probs = []

        for infer_var_hypo in infer_var.possible_values:
            prob_sum = 0.0
            var_dict[infer_var_name] = infer_var_hypo

            for comb in combo:
                prior_prob = 1.0
                for i, (val, prob) in enumerate(comb):
                    var_dict[left[i]] = val
                    prior_prob *= prob

                logits, individual_likelihoods, node_result = (
                    self.calculate_prob_product(var_dict, calc)
                )

                all_node_results.extend(node_result)

                prob_contribution = logits * prior_prob
                prob_sum += prob_contribution
            probs.append(prob_sum)

        probs = np.array(probs)
        probs = (probs / probs.sum()).tolist()

        return (
            probs,
            self.recorder,
            all_node_results,
        )
