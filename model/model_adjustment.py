import copy
import inspect
import os

import pandas as pd
from scipy.stats import entropy
import openai
import numpy as np
from termcolor import cprint

from BayesianInference import BayesianInferenceModel
import ElementExtractor

POSTERIOR_STABILITY_THRESHOLD = 0.01
BENEFIT_THRESHOLD = 0.02
ENTROPY_THRESHOLD = -0.673
EVIDENCE_PLATEAU_THRESHOLD = 0.01
EVIDENCE_THRESHOLD = -1.0
MODEL_SPACE = [
    ['State', 'Observation', 'Belief', 'Action', 'Goal'],   # POMDP
    ['State', 'Observation', 'Belief', 'Action'],           # POMDP variant without Goal
    ['State', 'Observation', 'Belief'],                     # Simple Markov Model
    ['State', 'Belief', 'Action', 'Goal'],                  # POMDP variant without Observation
    ['State', 'Belief', 'Action'],                          # POMDP variant without Observation and Goal
    ['State', 'Belief'],                                    # Simple Markov Model variant without Observation
    ['State', 'Action', 'Goal'],                            # MDP
    ['State', 'Action'],                                    # MDP variant without Goal
    ['State'],                                              # MDP variant without Action and Goal
]
ALL_VARIABLES = [
    "State",
    "Observation",
    "Belief",
    "Action",
    "Goal",
    "Utterance",
    "Belief of Goal"
]
LATENT_VARIABLES = [
    "Observation",
    "Belief",
    "Action",
    "Goal",
]


def subset_kwargs(func, kwargs):
    """
    Due to the number of arguments, we need to subset kwargs.

    TODO: Get rid of this function at some point once the code is refactored.
    """
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def tv_distance(p1: list, p2: list) -> float:
    """Compute TVD between two probability lists."""
    if len(p1) != len(p2):
        raise ValueError("Lists must have same length")
    p1_array = np.array(p1)
    p2_array = np.array(p2)
    # Normalize
    p1_array = p1_array / np.sum(p1_array) if np.sum(p1_array) > 0 else p1_array
    p2_array = p2_array / np.sum(p2_array) if np.sum(p2_array) > 0 else p2_array
    # Compute TVD
    tv_dist = 0.5 * np.sum(np.abs(p1_array - p2_array))
    return tv_dist


def model_discovery(
    curr_timestep,
    final_timestep,
    verbose,
    time_variables,
    previous_belief,
    inf_agent_name,
    inf_var_name,
    estimation_dictionary,
    infer_last_timestamp,
    no_observation_hypothesis,
    variable_values_with_time,
    all_probs,
    answerfunc,
    argmax,
    argmin,
    save_belief_probs,
    model_name,
    episode_name,
    infer_belief_at_timestamp,
    belief_name,
    get_variables_at_time,
    mmtom_get_variables_at_time,
    choices,
    K,
    llm,
    hypo_llm,
    hypo_method,
    full,
    preproposed_ob_hypos,
    last_state,
    inf_agent_action,
    model_graph,
    file_path,
    clear_current_nodes,
    eval_name,
    states,
    actions,
    question,
    precomputed_states,
    model_variables,
    no_model_adjustment,
    self,
    compute_response=False,
    observed_response_idx:int=None,
    observed_response_probs=None):
    """
    Args:
        observed_response_idx: The choice index of the observed response.
        compute_response: Whether to compute the response.
        model_graph: The complete model graph (i.e. V).
    """
    # TODO: This function has too many arguments, use locals to pass in kwargs
    kwargs = locals()
    # Whether to recompute the model
    recompute = False
    # Previous posterior for stability check of P(V' | R, X)
    prev_posterior = None
    # Only consider relevant latent variables for model experiments
    variables_for_experiments = [
        var for var in LATENT_VARIABLES
        if var not in model_graph[curr_timestep]
    ]
    # Compute P(V | X) given the current model
    bi_kwargs = subset_kwargs(bayesian_inference, kwargs)
    posterior, terminate, model_variables, log_evidence = bayesian_inference(**bi_kwargs)
    # Early stopping, no model adjustment needed
    if no_model_adjustment or terminate:
        return posterior, terminate, model_graph, model_variables
    # Otherwise, adjust the model to maximize score
    if compute_response:
        init_model_graph_score = log_evidence
        best_overall = (init_model_graph_score, posterior, copy.deepcopy(model_graph), model_variables)
        print(f"Init model graph score: {init_model_graph_score}")
    # Maximizing negative entropy
    else:
        # Use entropy for regular inference
        init_model_graph_score = -entropy(posterior)
    def get_model_score(model_graph, new_model_layer, model_variables):
        """
        Computes the score of a new model layer on the current timestep.

        Args:
            model_graph: The current, complete model graph.
            new_model_layer: The new model layer to add to the model.
            model_variables: The model variables.
        """
        # Copy the assigned models
        new_model_graph = copy.deepcopy(model_graph)
        new_model_graph[curr_timestep] = new_model_layer 
        # Clear hypotheses and nodes for current timestap from file
        clear_current_hypotheses(file_path, curr_timestep)
        clear_current_nodes(self, curr_timestep)
        # Recompute P(V | X) with the new model
        posterior, terminate, model_variables, log_evidence = bayesian_inference(
            curr_timestep,
            final_timestep,
            verbose,
            time_variables,
            previous_belief,
            inf_agent_name,
            inf_var_name,
            estimation_dictionary,
            infer_last_timestamp,
            no_observation_hypothesis,
            variable_values_with_time,
            all_probs,
            answerfunc,
            argmax,
            argmin,
            save_belief_probs,
            model_name,
            episode_name,
            infer_belief_at_timestamp,
            belief_name,
            get_variables_at_time,
            mmtom_get_variables_at_time,
            choices,
            K,
            llm,
            hypo_llm,
            hypo_method,
            full,
            preproposed_ob_hypos,
            last_state,
            inf_agent_action,
            new_model_graph,
            eval_name,
            states,
            actions,
            question,
            precomputed_states,
            model_variables,
            self,
            compute_response=compute_response,
            observed_response_idx=observed_response_idx,
            observed_response_probs=observed_response_probs)
        # Calculate new model fitness
        if compute_response:
            score = log_evidence# + np.log(posterior[observed_response_idx])
            return score, posterior
        else:
            # Use entropy for regular inference
            score = - entropy(posterior)
            return score, posterior
    # Explore new model layers
    while model_graph[curr_timestep] != MODEL_SPACE[0]:
        print(f"Current timestep: {curr_timestep}")
        print(f"Current model at timestep {curr_timestep}: {model_graph[curr_timestep]}")
        # All new model layer proposals are model_layer with 1 added variable
        model_layer_proposals = {}
        model_layer_proposals_posterior = {}
        # Add one variable at a time to the model and compute new model fitness
        for var in variables_for_experiments:
            new_model_layer = add_variable_to_model_layer(
                model_graph[curr_timestep], [var])
            # We constrain the models to be relevant inverse planning models
            if new_model_layer not in MODEL_SPACE:
                continue
            # Compute score for the new model graph with new layer
            score, posterior = get_model_score(
                model_graph, new_model_layer, model_variables)
            cprint(f"\tScore for adding {var} to {model_graph[curr_timestep]}: {score}", "red")
            model_layer_proposals[var] = score
            model_layer_proposals_posterior[var] = posterior
            # Update best-by-evidence cache for this candidate
            if compute_response:
                try:
                    if score > best_overall[0]:
                        candidate_graph = copy.deepcopy(model_graph)
                        candidate_graph[curr_timestep] = new_model_layer
                        best_overall = (score, posterior, candidate_graph, model_variables)
                except Exception:
                    pass
        # If we have no new model layer proposals, we're done
        if model_layer_proposals == {}:
            break
        # Otherwise, we need to recompute the model graph and clean the current 
        # hypotheses and nodes on file
        elif len(model_layer_proposals) > 1:
            recompute = True
        # Find the new_model_layer that has the highest score
        best_var, best_score = max(
            model_layer_proposals.items(), key=lambda item: item[1])
        best_posterior = model_layer_proposals_posterior[best_var]
        # Update posterior for next iteration's stability check
        if compute_response and best_var in model_layer_proposals:
            prev_posterior = best_posterior
        # Check posterior stability for evidence computation
        if compute_response and prev_posterior is not None:
            print(f"Prev posterior: {prev_posterior}")
            print(f"Current posterior: {best_posterior}")
            current_posterior = best_posterior
            tv_dist = tv_distance(prev_posterior, current_posterior)
            print(f"TV dist: {tv_dist}")
            # If posterior is stable, terminate
            if tv_dist < POSTERIOR_STABILITY_THRESHOLD:
                break
        # Check termination conditions
        if compute_response:
            print(f"Model graph: {model_graph}")
            print(f"Best var: {best_var}")
            print(f"Model layer proposals: {model_layer_proposals}")
            print(f"Best score: {best_score}, Init model graph score: {init_model_graph_score}")
            # For evidence computation, use evidence plateau + stability criteria
            evidence_improvement = best_score - init_model_graph_score
            print(f"Evidence improvement: {evidence_improvement}")
            # Check if evidence has plateaued and posterior is stable
            print(f"Prev posterior: {prev_posterior}")
            print(f"Current posterior: {best_posterior}")
            if evidence_improvement < EVIDENCE_PLATEAU_THRESHOLD:
                if prev_posterior is not None and best_var in model_layer_proposals:
                    current_posterior = best_posterior
                    print(f"Current posterior: {current_posterior}")
                    tv_dist = tv_distance(prev_posterior, current_posterior)
                    print(f"TV dist: {tv_dist}")
                    if tv_dist < POSTERIOR_STABILITY_THRESHOLD:
                        break
                else:
                    break    
            # Modify the model graph with the best variable
            model_graph[curr_timestep] = modify_variables(
                model_graph[curr_timestep], [best_var])
            init_model_graph_score = best_score
            breakpoint()
        else:
            # For regular inference, use existing thresholds
            if best_score > ENTROPY_THRESHOLD:
                model_graph[curr_timestep] = modify_variables(
                    model_graph[curr_timestep], [best_var])
                clear_current_hypotheses(file_path, curr_timestep)
                clear_current_nodes(self, curr_timestep)
                posterior, terminate, model_variables, log_evidence = bayesian_inference(
                    curr_timestep,
                    final_timestep,
                    verbose,
                    time_variables,
                    previous_belief,
                    inf_agent_name,
                    inf_var_name,
                    estimation_dictionary,
                    infer_last_timestamp,
                    no_observation_hypothesis,
                    variable_values_with_time,
                    all_probs,
                    answerfunc,
                    argmax,
                    argmin,
                    save_belief_probs,
                    model_name,
                    episode_name,
                    infer_belief_at_timestamp,
                    belief_name,
                    get_variables_at_time,
                    mmtom_get_variables_at_time,
                    choices,
                    K,
                    llm,
                    hypo_llm,
                    hypo_method,
                    full,
                    preproposed_ob_hypos,
                    last_state,
                    inf_agent_action,
                    model_graph,
                    eval_name,
                    states,
                    actions,
                    question,
                    precomputed_states,
                    model_variables,
                    self,
                    compute_response=compute_response,
                    observed_response_idx=observed_response_idx,
                    observed_response_probs=observed_response_probs)
                return posterior, terminate, model_graph, model_variables
            elif best_score - init_model_graph_score < BENEFIT_THRESHOLD:
                break
            else:
                model_graph[curr_timestep] = modify_variables(
                    model_graph[curr_timestep], [best_var])
                init_model_graph_score = best_score
        

    if recompute:
        # Clear hypotheses and nodes for current timestap from file
        clear_current_hypotheses(file_path, curr_timestep)
        clear_current_nodes(self, curr_timestep)
        # Recompute P(V | X)
        posterior, terminate, model_variables, log_evidence = bayesian_inference(
            curr_timestep,
            final_timestep,
            verbose,
            time_variables,
            previous_belief,
            inf_agent_name,
            inf_var_name,
            estimation_dictionary,
            infer_last_timestamp,
            no_observation_hypothesis,
            variable_values_with_time,
            all_probs,
            answerfunc,
            argmax,
            argmin,
            save_belief_probs,
            model_name,
            episode_name,
            infer_belief_at_timestamp,
            belief_name,
            get_variables_at_time,
            mmtom_get_variables_at_time,
            choices,
            K,
            llm,
            hypo_llm,
            hypo_method,
            full,
            preproposed_ob_hypos,
            last_state,
            inf_agent_action,
            model_graph,
            eval_name,
            states,
            actions,
            question,
            precomputed_states,
            model_variables,
            self,
            compute_response=compute_response,
            observed_response_idx=observed_response_idx,
            observed_response_probs=observed_response_probs)

    # Prefer best-by-evidence result over the last computed
    if compute_response:
        try:
            if best_overall is not None:
                best_score, best_post, best_graph, _best_vars = best_overall
                # Only print once at the outermost (final timestep) context if available
                try:
                    if curr_timestep == final_timestep - 1:
                        cprint(f"Returning best-by-evidence model with score {best_score}", "cyan")
                        cprint(f"Best model graph (by timestep): {best_graph}", "cyan")
                except Exception:
                    pass
                return best_post, False, best_graph, _best_vars
        except Exception:
            pass
    return posterior, terminate, model_graph, model_variables


def bayesian_inference(
    curr_timestep,
    final_timestep,
    verbose,
    time_variables,
    previous_belief,
    inf_agent_name,
    inf_var_name,
    estimation_dictionary,
    infer_last_timestamp,
    no_observation_hypothesis,
    variable_values_with_time,
    all_probs,
    answerfunc,
    argmax,
    argmin,
    save_belief_probs,
    model_name,
    episode_name,
    infer_belief_at_timestamp,
    belief_name,
    get_variables_at_time,
    mmtom_get_variables_at_time,
    choices,
    K,
    llm,
    hypo_llm,
    hypo_method,
    full,
    preproposed_ob_hypos,
    last_state,
    inf_agent_action,
    model_graph,
    eval_name,
    states,
    actions,
    question,
    precomputed_states,
    model_variables,
    self,
    compute_response=False,
    observed_response_idx=None,
    observed_response_probs=None,
): # -> tuple[list[float], bool, dict[int, dict[str, Variable]]]:
    verbose = False
    no_observation_hypothesis = "NONE"
    # Iterate over all timestamps
    for t in range(curr_timestep, final_timestep):
        cprint(f"Inferring model for timestamp {t}", "cyan")
        time_variables[t]["Previous Belief"] = previous_belief
        now_variables = []
        inf_name = f"{inf_agent_name}'s {inf_var_name}"
        variable_types = []
        for var_name in model_graph[t]:
            variable_types.append((inf_agent_name, var_name))
        variable_types.append(("", "All Actions"))
        # propose hypotheses for new model layer at current timestep
        if t == curr_timestep:
            if "MMToM" not in eval_name:
                curr_time_vars, preproposed_ob_hypos, last_state, inf_agent_action = get_variables_at_time(
                    time=t,
                    vals=variable_values_with_time[t],
                    variable_types=variable_types,
                    inf_agent_name=inf_agent_name,
                    inf_var_name=inf_var_name,
                    choices=choices,
                    K=K,
                    llm=llm,
                    hypo_llm=hypo_llm,
                    verbose=verbose,
                    hypo_method=hypo_method,
                    full=full,
                    preproposed_ob_hypos=preproposed_ob_hypos,
                    last_state=last_state,
                    inf_agent_action=inf_agent_action,
                    eval_name=eval_name,
                    precomputed_states=precomputed_states,
                )
            else:
                curr_time_vars, preproposed_ob_hypos = mmtom_get_variables_at_time(
                    time=t,
                    variable_types=variable_types,
                    inf_agent_name=inf_agent_name,
                    inf_var_name=inf_var_name,
                    choices=choices,
                    K=K,
                    llm=llm,
                    hypo_llm=hypo_llm,
                    verbose=verbose,
                    hypo_method=hypo_method,
                    full=full,
                    preproposed_ob_hypos=preproposed_ob_hypos,
                    states=states,
                    actions=actions,
                    question=question,
                )
            model_variables[t] = curr_time_vars
        else: # other timestamp we use the previous model
            curr_time_vars = model_variables[t]
        # Get the variables for the current timestamp
        variables = curr_time_vars
        if t == final_timestep - 1:
            # Use evidence computation instead of regular inference
            if compute_response:
                # Set up the model for evidence computation
                all_variables = []
                for var_name in model_graph[t]:
                    # Find the corresponding agent-prefixed variable name
                    _inf_name = f"{inf_agent_name}'s {var_name}" if var_name != "State" else var_name
                    if _inf_name in variables:
                        all_variables.append(variables[_inf_name])
                # Add Response variable if not present
                if "Response" not in [v.name for v in all_variables]:
                    response_var = ElementExtractor.Variable(
                        name="Response",
                        in_model=True,
                        is_observed=True,
                        possible_values=[choices[observed_response_idx]],
                        prior_probs=[1.0]
                    )
                    all_variables.append(response_var)
                # Create BayesianInferenceModel instance
                bim = BayesianInferenceModel(
                    variables=all_variables,
                    context=question,  # Use question as context
                    llm=llm,
                    verbose=verbose,
                    inf_agent=inf_agent_name,
                    model_name=model_name,
                    eval_name=eval_name,
                    episode_name=episode_name,
                    answer_choices=choices,
                    K=K,
                    all_prob_estimations=estimation_dictionary,
                    no_observation_hypothesis=no_observation_hypothesis,
                    reduce_hypotheses=self.reduce_hypotheses if hasattr(
                        self, 'reduce_hypotheses') else False,
                )
                # Compute evidence
                log_evidence, posterior, node_posterior, latent_var_names, calc, response_likelihoods, latent_var_combinations = bim.compute_evidence(
                    observed_response_idx=observed_response_idx,
                    model_name=model_name,
                    episode_name=episode_name,
                    observed_response_probs=observed_response_probs
                )
                if posterior:
                    # Full marginalization: P(R|X) = Σ_V P(R|V,X) × P(V|X)
                    answer_choice_probs = []
                    for choice in choices:
                        choice_prob = 0.0  # P(R=choice|X)
                        # Sum over all latent variable combinations V
                        for latent_id, latent_combo in enumerate(latent_var_combinations):
                            # Set up variable dictionary for this combination V
                            var_dict = {}
                            for i, (val, prob) in enumerate(latent_combo):
                                var_dict[latent_var_names[i]] = val
                            # Set Response to current choice
                            var_dict["Response"] = choice
                            # Use the posterior weight from compute_evidence via latent_id
                            posterior_weight = posterior.get(latent_id, 0.0)
                            # breakpoint()
                            # Compute P(R=choice|V,X)
                            likelihood = bim.calculate_likelihood_given_variables(var_dict, calc)
                            # Add contribution: P(R=choice|V,X) × P(V|X)
                            choice_prob += likelihood * posterior_weight
                        answer_choice_probs.append(choice_prob)
                    # Normalize to get proper probabilities
                    total_prob = sum(answer_choice_probs)
                    if total_prob > 0:
                        answer_choice_probs = [p / total_prob for p in answer_choice_probs]
                    else:
                        answer_choice_probs = [1.0 / len(choices)] * len(choices)
                    posterior = answer_choice_probs
                    print(f"P(R|X) marginalized: {posterior} at timestep {t}")
                    print(f"log_evidence: {log_evidence}")
                else:
                    # Fallback to uniform distribution
                    posterior = [1.0 / len(choices)] * len(choices)
                # Evidence computation doesn't use termination logic
                terminate = False
            # Regular inference
            else:
                posterior, estimation_dictionary, all_probs = infer_last_timestamp(
                    self,
                    t=t,
                    time_variables=variables,
                    inf_name=inf_name,
                    inf_var_name=inf_var_name,
                    now_variables=now_variables,
                    no_observation_hypothesis=no_observation_hypothesis,
                    var_vals_with_time=variable_values_with_time,
                    all_probs=all_probs,
                    all_prob_estimations=estimation_dictionary,
                )
                # determine if we can stop inference
                posterior = list(posterior)
                terminate = False
                # score = 1 / entropy(posterior)
                score = - entropy(posterior)
                if len(posterior) == 2:
                    # if max(posterior) > terminate_threshold:
                    if score > ENTROPY_THRESHOLD:
                        terminate = True
                else:
                    # if answerfunc == argmax and max(posterior) > terminate_threshold:
                    if answerfunc == argmax and score > ENTROPY_THRESHOLD:
                        terminate = True
                    elif answerfunc == argmin and min(posterior) < 0.2:
                        terminate = True
                if terminate is True:
                    save_belief_probs(all_probs, model_name, episode_name)
            return posterior, terminate, model_variables, None
        else:
            previous_belief, estimation_dictionary, all_probs = infer_belief_at_timestamp(
                self,
                t=t,
                time_variables=variables,
                previous_belief=previous_belief, 
                belief_name=belief_name, 
                variable_values_with_time=variable_values_with_time,
                all_probs=all_probs, 
                no_observation_hypothesis=no_observation_hypothesis, 
                all_prob_estimations=estimation_dictionary)


def gpt_call(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def initial_model_proposal(question, query_variable, contains_utterance):
    """
    Proposes an initial set of variables to use for the agent model.

    query_variable is the latent variable to query from the question, e.g.
    'Belief', 'State', etc.

    Args:
        question: The question being solved from a specific dataset.
        query_variable: The latent variable to query from the question.
        contains_utterance: Whether the question contains an utterance.

    Returns:
        assigned_model: The initial set of variables to use for the agent model.
    """
    example_question = (
        "Sally has a ball. She puts it in her basket. ",
        "When Sally goes out for a walk, ",
        "Anne moves the ball out of the basket and puts it in the box. ",
        "Where will Sally look for the ball?"
    )
    example_answer = "['State', 'Observation', 'Belief']"
    prompt = (
        "What variables are necessary to solve this question? "
        "Please provide the answer without an explanation. "
        "Please select from the following: "
        "['State', 'Observation', 'Belief', 'Action', 'Goal'] "
        "State: The true condition of the environment. "
        "This should always be included. "
        "Observation: The observed information about the state. "
        "Include this when the agent has partial observations of the state."
        "Belief: The agent's current estimation of the true state based on "
        "the state or its observation."
        "Action: A move made by the agent, informed by the state or belief. "
        "Include this only when it is directly relevant to answering the question. "
        "Goal: The objective the agent is trying to achieve. Include this "
        "only if 'Action' is included. \n"
        f"Question: {example_question} \n"
        f"Variables: {example_answer} \n"
        f"Question: {question} \n"
        "Variables: "
        )
    # Grab LM response and evaluate to list of variables
    response = gpt_call(prompt)
    assigned_model = eval(response)
    print(f"Assigned model: {assigned_model}")
    # If the query variable is not in the initial agent model, add it
    if query_variable not in assigned_model:
        assigned_model.append(query_variable)
        assigned_model = sorted(assigned_model, key=ALL_VARIABLES.index)
    # NOTE: This might not be useful in the event we're computing P(r|V,X)
    if assigned_model not in MODEL_SPACE:
        if "State" not in assigned_model:
            assigned_model.append("State")
        if "Observation" in assigned_model and "Belief" not in assigned_model:
            assigned_model.append("Belief")
        if "Goal" in assigned_model and "Action" not in assigned_model:
            assigned_model.append("Action")
        assigned_model = sorted(assigned_model, key=ALL_VARIABLES.index)
    # assert assigned_model in model_space
    temp_assigned_model = [
        model for model in assigned_model if model != "Belief of Goal"]
    assert temp_assigned_model in MODEL_SPACE
    if contains_utterance:
        assigned_model.append("Utterance")
    return assigned_model


def determine_realistic_questions(question):
    prompt = f"Determine whether the question is about realistic physical states. \
        It is about physical world states if it involves no agent. \n\
        Please respond with 'Yes' or 'No'. \n\
        If the answer is 'Yes', the question often ends with 'Where is A really?'. Otherwise, respond 'No'. \n\
        Question: Where does Emily think Jack will look for the celery? \
        Realistic: No \n\
        Question: Where will Jack look for the celery? \
        Realistic: No \n\
        Question: Where was the celery initially? \
        Realistic: No \n\
        Question: Where is the celery really? \
        Realistic: Yes \n\
        Question: {question} \
        Realistic: "
    response = gpt_call(prompt)
    if response == "Yes":
        return True
    else:
        return False


def determine_memory_questions(question):
    prompt = f"Determine whether the question is about where an object was initially. \
        Please respond with 'Yes' or 'No'. \n\
        If the answer is 'Yes', the question often ends with 'Where was A initially?'. Otherwise, respond 'No'. \n\
        Question: Where does Emily think Jack will look for the celery? \
        Response: No \n\
        Question: Where will Jack look for the celery? \
        Response: No \n\
        Question: Where is the celery really? \
        Response: No \n\
        Question: Where was the celery initially? \
        Response: Yes \n\
        Question: {question} \
        Response: "
    response = gpt_call(prompt)
    if response == "Yes":
        return True
    else:
        return False


def determine_higher_order_belief(question):
    prompt = f"Determine whether the question is about a higher-order belief. \
        A higher-order belief refers to a belief about another person's belief, goal or action. \n\
        It is not a high-order belief if it only asks about one agent's belief. \n\
        Please respond with 'Yes' or 'No'. \n\
        If the answer is 'Yes', the question often ends with 'Where does A think that B ...?' inwhich A and B are two agents. Otherwise, respond 'No'. \n\
        When the story only have one person, respond with 'No'. \n\
        Question: [A story involving several people.] Where will Jack look for the celery? \
        Higher-order belief: No \n\
        Question: [A story involving several people.] Where does Jack think that Chloe searches for the hat? \
        Higher-order belief: Yes \n\
        When the story only have one person, respond with 'No'. \n\
        Question: {question} \
        Higher-order belief: "
    response = gpt_call(prompt)
    if response == "Yes":
        return True
    else:
        return False


def modify_variables(assigned_model, vars, action="add"):
    modified_model = list(assigned_model)
    if action == "add":
        if vars == ["Action"]:
            vars.append("Goal")
        for var in vars:
            assert var not in modified_model
            modified_model.append(var)
        modified_model = sorted(
            modified_model, key=ALL_VARIABLES.index
        )
    elif action == "remove":
        for var in vars:
            assert var in modified_model
            modified_model.remove(var)
    
    return modified_model


def add_variable_to_model_layer(model_layer, vars):
    """
    Adds a variable to a model at a specific timestep.

    The model is a list of variables at a specific timestep.

    Args:
        assigned_model: The model at a specific timestep.
        vars: The variables to add to the model layer.

    Returns:
        The modified model.
    """
    modified_model = list(model_layer)
    if vars == ["Action"]:
        vars.append("Goal")
    for var in vars:
        assert var not in modified_model
        modified_model.append(var)
    modified_model = sorted(
        modified_model, key=ALL_VARIABLES.index
    )
    return modified_model


def clear_current_hypotheses(file_path, time):
    # Load the data
    # TODO: Determine why this would fail at all
    if not os.path.exists(file_path):
        return
    df = pd.read_csv(file_path)
    df = df[df['Time'] != time]
    df.to_csv(file_path, index=False)
    return
