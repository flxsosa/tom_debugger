"""
TimestepInference Module: Core temporal inference functions for AutoToM.

This module handles Bayesian inference at individual timesteps, managing belief updates
and final variable inference across the temporal sequence of a Theory of Mind scenario.
"""

from ElementExtractor import *
from utils import *
from BayesianInference import BayesianInferenceModel

def infer_belief_at_timestamp(
    self,
    time_variables,
    i,
    previous_belief,
    belief_name,
    variable_values_with_time,
    all_probs,
    no_observation_hypothesis,
    all_prob_estimations,
):
    """
    Infer belief state at timestep i using Bayesian inference.
    
    This function performs belief updating for intermediate timesteps (not the final one).
    It constructs a Bayesian network from available variables and infers the agent's
    belief state, which becomes the "Previous Belief" for the next timestep.
    
    Args:
        self: ProblemSolver instance containing configuration
        time_variables: Dict/list of variables available at timestep i
        i: Current timestep index
        previous_belief: Belief state from previous timestep (used as prior)
        belief_name: Name of the belief variable to infer
        variable_values_with_time: Optional temporal variable values
        all_probs: Accumulated probability results across timesteps
        no_observation_hypothesis: Fallback value when no observations available
        all_prob_estimations: Cache of previously computed likelihoods
        
    Returns:
        Tuple of (updated_previous_belief, updated_prob_estimations, updated_all_probs)
    """
    
    # Extract variables for current timestep i
    if isinstance(time_variables, list):
        var_i = time_variables[i]  # List format: get variables at index i
    elif isinstance(time_variables, dict):  # Dict format: variables already at specific timestep
        var_i = time_variables
    
    # Build variable list for Bayesian inference, excluding metadata variables
    now_variables = []
    for key, item in var_i.items():
        if (
            key != belief_name  # Don't include belief we're inferring
            and key != "All Actions"  # Metadata variable
            and key != "Ground Truth State"  # Metadata variable
        ):
            now_variables.append(item)
    
    # Add the belief variable we want to infer
    if belief_name in var_i:
        now_variables.append(var_i[belief_name])
    
    # Set context based on dataset type
    if "BigToM" in self.dataset_name:
        context = self.story  # BigToM uses full story as context
    else:
        context = ""  # Other datasets use minimal context
    
    # Create Bayesian inference model for this timestep
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
    )

    # Perform Bayesian inference to get belief probabilities
    try:
        results, all_prob_estimations, all_node_results = inference_model.infer(
            "Belief", self.model_name, self.episode_name, self.init_belief
        )
    except Exception as e:
        print(f"Exception {e}")
        # Return fallback belief if inference fails
        return (
            Variable("Previous Belief", True, False, ["NONE"], np.ones(1)), 
            all_prob_estimations, 
            all_probs
        )
    
    # Record node-level results for analysis/debugging
    self.translate_and_add_node_results(self, i, all_node_results)
    
    # Create updated previous belief for next timestep
    previous_belief = deepcopy(var_i[belief_name])
    previous_belief.prior_probs = np.array(results)  # Set computed probabilities as priors
    previous_belief.name = "Previous Belief"  # Rename for next timestep
    
    # Log belief update
    enh_print(
        f"After time {i}: Belief Probs Updated to {previous_belief.possible_values}, {results}"
    )
    
    # Extract chunk information for temporal tracking
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]
    
    # Store probability results for this timestep
    now_probs = {
        "Chunk": chunk,
        f"Probs({self.choices})": results,
    }
    all_probs.append(now_probs)
    
    return (previous_belief, all_prob_estimations, all_probs)

def infer_last_timestamp(
    self,
    time_variables,
    i,
    inf_name,
    inf_var_name,
    now_variables,
    no_observation_hypothesis,
    variable_values_with_time,
    all_probs,
    all_prob_estimations,
):
    """
    Infer target variable at the final timestep using Bayesian inference.
    
    This function performs the final inference step, computing probabilities for the
    target variable (e.g., Belief, Action, Goal) that answers the ToM query. Unlike
    intermediate timesteps, this focuses on the specific variable of interest rather
    than belief updating.
    
    Args:
        self: ProblemSolver instance containing configuration
        time_variables: Dict/list of variables available at final timestep
        i: Final timestep index
        inf_name: Name of the variable to infer (e.g., "Belief", "Action")
        inf_var_name: Specific variable name for inference (usually same as inf_name)
        now_variables: Pre-populated list of variables for this timestep
        no_observation_hypothesis: Fallback value when no observations available
        variable_values_with_time: Optional temporal variable values
        all_probs: Accumulated probability results across timesteps
        all_prob_estimations: Cache of previously computed likelihoods
        
    Returns:
        Tuple of (final_probabilities, updated_prob_estimations, updated_all_probs)
    """
    
    # Extract variables for final timestep i
    if isinstance(time_variables, list):
        var_i = time_variables[i]  # List format: get variables at index i
    elif isinstance(time_variables, dict):  # Dict format: variables already at specific timestep
        var_i = time_variables
    
    # Build final variable list, excluding metadata and the target variable
    for key, item in var_i.items():
        if key != inf_name and key != "All Actions" and key != "Ground Truth State":
            if item != "NONE":  # Only include non-empty variables
                item.name = key  # Ensure proper variable naming
                now_variables.append(item)

    # Add the target variable we want to infer
    try:
        now_variables.append(var_i[inf_name])
    except Exception:
        print(f"No {inf_name}!")
        return None

    # Debug output for variable selection
    if self.verbose:
        print("chosen variables: \n\n\n", now_variables, "\n\n\n")

    # Set context based on dataset type
    if "BigToM" in self.dataset_name:
        context = self.story  # BigToM uses full story as context
    else:
        context = ""  # Other datasets use minimal context
    
    # Create Bayesian inference model for final timestep
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
    )

    # Perform final Bayesian inference on target variable
    results, all_prob_estimations, all_node_results = inference_model.infer(
        inf_var_name, self.model_name, self.episode_name, self.init_belief
    )
    
    # Record node-level results for analysis/debugging
    self.translate_and_add_node_results(self, i, all_node_results)
    
    # Save intermediate results to disk for analysis
    save_node_results(
        self.intermediate_node_results,
        self.model_name,
        self.episode_name,
        self.back_inference,
        self.reduce_hypotheses,
    )
    
    # Log final inference results
    enh_print(
        f"After time {i}: {inf_name} Probs calculated as {var_i[inf_name].possible_values}, {results}"
    )
    
    # Extract chunk information for temporal tracking
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]
    
    # Store final probability results
    now_probs = {
        "Chunk": chunk,
        f"Probs({self.choices})": results,
    }
    all_probs.append(now_probs)
    
    return results, all_prob_estimations, all_probs
