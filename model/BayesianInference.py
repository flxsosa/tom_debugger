import itertools
from copy import deepcopy, copy

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        eval_name,
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
        self.eval_name = eval_name
        self.answer_choices = answer_choices
        new_variables = []
        for i, var in enumerate(variables):
            if isinstance(var, list):
                if isinstance(var[0], float):
                    continue
            if verbose:
                print(var)
            if "'s " in var.name:
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
                observed_var_names = son_graph[now_var] if now_var in son_graph else []
                for j in observed_var_names:
                    new_parent_graph[j].remove(now_var)
                    for i in left:
                        if i not in new_parent_graph[j]:
                            new_parent_graph[j].append(i)
                key_list = list(new_parent_graph.keys())
                for key in key_list:
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
                    info_var.append(
                        f"{self.inf_agent}'s {parent}: {var_dict[parent]}")

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
                    self.eval_name == "BigToM_fatb"
                    or self.eval_name == "BigToM_fafb"
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

            self.recorder[key] = logits

            node_results_tracker.append((son, parents, copy(var_dict), logits))
            individual_likelihoods[(son, tuple(parents), var_dict[son])] = logits

            prob *= logits

        return prob, individual_likelihoods, node_results_tracker

    def reduce_obs_hypospace(self):
        """Reduce the hypothesis space for the observation variable.

        Args:
            None
        """
        all_node_results = []
        # Gather hypotheses for observation variable
        observation_distribution = self.variables["Observation"].possible_values
        # Get the value of the state
        state_value = self.variables["State"].possible_values[0]
        var_dict = {}
        var_dict["State"] = state_value
        po = []
        for obs_hypothesis in observation_distribution:
            var_dict["Observation"] = obs_hypothesis
            p, ind_lh, node_result = self.calculate_prob_product(
                var_dict, [("Observation", ["State"])]
            )
            all_node_results.extend(node_result)
            po.append(p)
        new_hypos = []
        for i, obs_hypothesis in enumerate(observation_distribution):
            # For a possible observation hypotheses,
            # if the likelihood P(o_i|s) is lower than 0.03
            # we discard it
            if po[i] <= 0.03:
                continue
            new_hypos.append(obs_hypothesis)
        if len(new_hypos) == 0:
            new_hypos = observation_distribution
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
        if ("BigToM" in self.eval_name and
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
        latent_var_names = []
        observed_var_names = []
        # Add observed variables to right, latent variables to left
        for var in self.variables.values():
            if var.is_observed:
                observed_var_names.append(var.name)
            else:
                latent_var_names.append(var.name)
        # NOTE: This is where we print the query we are inferring
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
            if "Observation" in latent_var_names:
                all_node_results += self.reduce_obs_hypospace()
            if "Belief" in latent_var_names and infer_var_name != "Belief":
                all_node_results += self.reduce_belief_hypospace()
        # Generate all possible combinations of latent variable values
        combo = self.recompute_combinations(latent_var_names, infer_var_name)
        var_dict = {}
        # Set observed variables to their possible values
        for ob_var_name in observed_var_names:
            var_dict[ob_var_name] = self.variables[ob_var_name].possible_values[0]
        all_var_names = latent_var_names + observed_var_names
        # List of tuples (variable name, list of parent variables)
        calc = []
        for var_name in all_var_names:
            if var_name in self.parent_graph:
                need_compute = False
                if var_name in latent_var_names:
                    need_compute = True
                for par_var_name in self.parent_graph[var_name]:
                    if par_var_name in latent_var_names:
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
                    var_dict[latent_var_names[i]] = val
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

    def compute_evidence(self, observed_response_idx, model_name, episode_name,observed_response_probs):
        """
        Compute evidence and posterior for cognitive debugging.
        
        Mathematical steps:
        1. Compute P(V|X) using actual variable priors
        2. Compute P(R|V,X) for observed response R
        3. Compute P(V|R,X) ∝ P(R|V,X) × P(V|X) using Bayes' rule
        4. Find V* = argmax_V P(V|R,X) 
        5. Compute P(R|V*,X) for all possible responses R
        """
        self.rewrite_graph()
        latent_var_names = []
        observed_var_names = []
        # Sort through variables of the current model
        for key, var in self.variables.items():
            if var.is_observed:
                observed_var_names.append(var.name)
            else:
                latent_var_names.append(var.name)
        # Handle hypothesis reduction
        all_node_results = []
        if self.reduce_hypotheses:
            if "Observation" in latent_var_names:
                all_node_results += self.reduce_obs_hypospace()
            if "Belief" in latent_var_names:
                all_node_results += self.reduce_belief_hypospace()
        # Generate combinations
        latent_var_combinations = self.recompute_combinations_for_evidence(latent_var_names)
        # Set up variable dictionary
        var_dict = {}
        for observed_var_name in observed_var_names:
            var_dict[observed_var_name] = self.variables[observed_var_name].possible_values[0]
        
        # Build calculation dependencies
        all_var_names = latent_var_names + observed_var_names
        calc = []
        for var_name in all_var_names:
            if var_name in self.parent_graph:
                need_compute = False
                if var_name in latent_var_names:
                    need_compute = True
                for par_var_name in self.parent_graph[var_name]:
                    if par_var_name in latent_var_names:
                        need_compute = True
                if not need_compute:
                    continue
                calc.append((var_name, self.parent_graph[var_name]))
                # print(f'P({var_name}|{",".join(self.parent_graph[var_name])})', end="")
        
        # Ensure Response variable is in calculation dependencies
        if "Response" in self.variables:
            response_in_calc = any(item[0] == "Response" for item in calc)
            if not response_in_calc:
                if "Response" in self.parent_graph:
                    calc.append(("Response", self.parent_graph["Response"]))
                else:
                    response_deps = []
                    for var_name in all_var_names:
                        if var_name in ["Belief", "Goal", "State", "Action", "Observation"]:
                            response_deps.append(var_name)
                    if response_deps:
                        calc.append(("Response", response_deps))
        
        # Step 2: Compute weights for soft (for evidence) and hard (for V* selection)
        evidence_sum_soft = 0.0  # Σ_V (Σ_r q[r] P(r|V,X)) P(V|X)
        evidence_sum_hard = 0.0  # Σ_V P(R*|V,X) P(V|X)
        posterior_weights_soft = {}  # latent_id -> soft weight
        posterior_weights_hard = {}  # latent_id -> hard weight
        
        for latent_id, latent_combo in enumerate(latent_var_combinations):
            # Set up variable dictionary for this combination V
            for i, (val, prob) in enumerate(latent_combo):
                var_dict[latent_var_names[i]] = val

            # Compute P(V|X) using actual variable priors
            prior_prob = 1.0
            for i, var_name in enumerate(latent_var_names):
                var = self.variables[var_name]
                if var.prior_probs is not None:
                    val_idx = var.possible_values.index(var_dict[var_name])
                    prior_prob *= var.prior_probs[val_idx]
                else:
                    prior_prob *= (1.0 / len(var.possible_values))

            # Compute soft and hard likelihoods
            if observed_response_probs is not None:
                # Soft: E_R~q[R][P(R|V,X)]
                likelihood_soft = 0.0
                for p_r, choice in zip(observed_response_probs, self.answer_choices):
                    var_dict["Response"] = choice
                    lr = self.calculate_likelihood_given_variables(var_dict, calc)
                    likelihood_soft += p_r * lr
                # Hard: P(R*|V,X)
                var_dict["Response"] = self.answer_choices[observed_response_idx]
                likelihood_hard = self.calculate_likelihood_given_variables(var_dict, calc)
            else:
                var_dict["Response"] = self.answer_choices[observed_response_idx]
                likelihood_hard = self.calculate_likelihood_given_variables(var_dict, calc)
                likelihood_soft = likelihood_hard

            # Accumulate contributions
            contrib_soft = likelihood_soft * prior_prob
            contrib_hard = likelihood_hard * prior_prob
            evidence_sum_soft += contrib_soft
            evidence_sum_hard += contrib_hard
            posterior_weights_soft[latent_id] = posterior_weights_soft.get(latent_id, 0.0) + contrib_soft
            posterior_weights_hard[latent_id] = posterior_weights_hard.get(latent_id, 0.0) + contrib_hard
        
        # Step 3: Normalize posteriors
        log_evidence = np.log(evidence_sum_soft) if evidence_sum_soft > 0 else float('-inf')
        posterior = {}
        for latent_id, weight in posterior_weights_soft.items():
            posterior[latent_id] = weight / evidence_sum_soft if evidence_sum_soft > 0 else 0.0
        posterior_hard = {}
        for latent_id, weight in posterior_weights_hard.items():
            posterior_hard[latent_id] = weight / evidence_sum_hard if evidence_sum_hard > 0 else 0.0
        # Debug: summarize posterior over V (top-3 combos) and aggregate by Belief if present
        # NOTE: We're only printing the top 3 combinations for now
        try:
            if len(posterior) > 0:
                # Top-k by probability
                topk = sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:3]
                print("Top-V combinations (latent_id -> prob):")
                for lid, v in topk:
                    print(f"  {lid} -> {v:.4f}")
                print("Top-V combinations (human -> prob):")
                for lid, v in topk:
                    combo = latent_var_combinations[lid]
                    human = tuple(val for (val, _p) in combo)
                    print(f"  {human} -> {v:.4f}")
                # Aggregate by Belief if exists among latent_var_names
                if "Belief" in latent_var_names:
                    agg = {}
                    for lid, v in posterior.items():
                        combo = latent_var_combinations[lid]
                        # find Belief value in this combo
                        belief_idx = latent_var_names.index("Belief")
                        belief_val = combo[belief_idx][0]
                        agg[belief_val] = agg.get(belief_val, 0.0) + v
                    print(f"latent_var_names: {latent_var_names}")
                    print(f"sum P(V|X; soft): {sum(posterior.values()):.4f}")
                    print(f"sum P(V|R*,X; hard): {sum(posterior_hard.values()):.4f}")
                    total_belief = sum(agg.values())
                    print("Aggregate P(Belief|X; soft):")
                    if abs(total_belief - 1.0) <= 1e-3:
                        for b, v in sorted(agg.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {b} -> {v:.4f}")
                    else:
                        # Print normalized view for readability
                        if total_belief > 0:
                            norm_agg = {b: v/total_belief for b, v in agg.items()}
                        else:
                            norm_agg = agg
                        for b, v in sorted(norm_agg.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {b} -> {v:.4f}")
                        print(f"[Warn] Belief marginal captured {total_belief:.4f} of mass; normalized for display.")
        except Exception:
            pass
        
        # Step 4: Find MAP estimate V* = argmax_V P(V|R*,X) using hard posterior
        if posterior_hard:
            v_map_id = max(posterior_hard.items(), key=lambda x: x[1])[0]
            
            # Step 5: Compute P(R|V*,X) for all possible responses R
            var_dict_map = {}
            # Reconstruct V* assignment from latent_var_combinations
            v_map_combo = latent_var_combinations[v_map_id]
            for i, var_name in enumerate(latent_var_names):
                var_dict_map[var_name] = v_map_combo[i][0]
            
            response_likelihoods = []
            for choice in self.answer_choices:
                var_dict_map["Response"] = choice
                likelihood = self.calculate_likelihood_given_variables(var_dict_map, calc)
                response_likelihoods.append(likelihood)
            
            # Normalize response likelihoods
            response_likelihoods = np.array(response_likelihoods)
            response_likelihoods = (response_likelihoods / response_likelihoods.sum()).tolist()
        else:
            response_likelihoods = [1.0 / len(self.answer_choices)] * len(self.answer_choices)
        print(f"P(R|V*,X): {response_likelihoods}")
        
        if self.verbose:
            print(f"\nEvidence Z(M) = {evidence_sum_soft:.6f}, log Z(M) = {log_evidence:.6f}")
        
        return (
            log_evidence,
            posterior,
            all_node_results,
            latent_var_names,
            calc,
            response_likelihoods,
            latent_var_combinations
        )

    def calculate_likelihood_given_variables(self, var_dict, calc):
        """
        Calculate P(R|V,X) - likelihood of response given latent variables.
        
        This is different from calculate_prob_product which computes P(V,R|X).
        """
        likelihood = 1.0
        for var_name, parents in calc:
            # Only compute likelihood for the Response variable
            if var_name == "Response":
                info_var = []
                for parent in parents:
                    # Map parent names to actual variable names in var_dict
                    if parent == "State":
                        # State is always just "State" in var_dict
                        if "State" in var_dict:
                            info_var.append(f"{parent}: {var_dict['State']}")
                    elif parent in var_dict:
                        # Other variables use their actual names
                        info_var.append(f"{self.inf_agent}'s {parent}: {var_dict[parent]}")
                    else:
                        # Skip if parent not found in var_dict
                        continue
                info = "\n".join(info_var)
                key = info + ";" + str(var_dict[var_name])
                if key in self.recorder:
                    likelihood = self.recorder[key]
                else:
                    likelihood = probs.get_likelihood(
                        info,
                        str(var_dict[var_name]),
                        model=self.llm,
                        verbose=self.verbose,
                        variable=var_name,
                        inf_agent=self.inf_agent,
                    )
                    self.recorder[key] = likelihood
                break  # Only compute for Response variable
        return likelihood

    def recompute_combinations_for_evidence(self, left):
        """Generate all possible combinations of latent variable values.
        
        Used for evidence computation.
        
        Args:
            left: List of latent variable names
            
        Returns:
            List of all possible combinations of latent variable values
        """
        combo = []
        for unob_var_name in left:
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

    def visualize_current_model(self, save_path=None):
        """Visualize `the current model graph structure."""
        
        G = nx.DiGraph()
        
        # Add nodes
        for var_name, var in self.variables.items():
            G.add_node(var_name, 
                    is_observed=var.is_observed,
                    possible_values=len(var.possible_values))
        
        # Add edges from parent_graph
        for child, parents in self.parent_graph.items():
            if child in self.variables:
                for parent in parents:
                    if parent in self.variables:
                        G.add_edge(parent, child)
        
        # Color nodes based on type
        colors = []
        for node in G.nodes():
            if node == 'State':
                colors.append('green')
            elif node == 'Observation':
                colors.append('yellow')
            elif node == 'Belief':
                colors.append('lightblue')
            elif node == 'Action':
                colors.append('orange')
            elif node == 'Goal':
                colors.append('darkblue')
            else:
                colors.append('gray')
        
        # Draw
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=colors, with_labels=True, 
                node_size=2000, font_size=12, font_weight='bold')
        
        plt.title(f"Model Graph: {self.model_name}")
        if save_path:
            plt.savefig(save_path)
        plt.show()