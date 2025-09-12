"""ProbSolver Module - Theory of Mind Problem Solver

This module implements a Bayesian inference-based system for solving Theory
of Mind (ToM) problems.

It can operate in two modes:
1. Automated mode: Dynamically discovers which mental state variables to track
2. Manual mode: Uses predefined variable sets for Bayesian inference

The system processes stories, extracts temporal mental state variables, and
uses Bayesian inference to answer questions about characters' beliefs, goals,
and actions.

High-level workflow:
1. Parse story and question to identify agents and inference targets
2. Extract temporal variables (states, actions, beliefs, etc.) from the story
3. For nested reasoning: reconstruct story from different agents' perspectives
4. Apply Bayesian inference to determine probabilities for each answer choice
5. Select the most likely answer based on computed probabilities

Key components:
- ProblemSolver: Main class that orchestrates the solving process
- information_extraction(): Extracts temporal mental state variables
- solve_with_automated_model(): Handles dynamic model discovery
- solve(): Main solving logic for both automated and manual modes
- main(): Entry point that processes datasets and evaluates performance
"""

import argparse
from copy import deepcopy
import time
import ast

from scipy.stats import entropy
import numpy as np

import model_adjustment
import Timeline
import utils
import DataLoader
import probs
import NodeResultTracker
import ElementExtractor
import TimestepInference
import ProblemParser
import Nested


def argmax(lst):
    """Find the index of the maximum value in a list."""
    return lst.index(max(lst))


def argmin(lst):
    """Find the index of the minimum value in a list."""
    return lst.index(min(lst))


class ProblemSolver:
    """
    Main class for solving Theory of Mind problems using Bayesian inference.
    
    Orchestrates the complete problem-solving pipeline from story parsing to 
    answer selection. Supports both automated model discovery and manual 
    variable specification modes.
    """

    def __init__(
        self,
        story,
        question,
        choices,
        K,
        assigned_model=None,
        model_name="sobag",
        episode_name="",
        llm="gpt-4o",
        hypo_llm="gpt-4o",
        verbose=False,
        dataset_name=None,
        hypo_method=None,
        nested=False,
        tab="",
        full=True,
        video_id=None,
        answerfunc=None,
        back_inference=False,
        reduce_hypotheses=False,
        precomputed_states=None,
        precomputed_actions=None,
        prev_hyp=None,
        no_model_adjustment=False,
        recursion_depth=0,
        nested_timeline_before=None,
        nested_time_variables_before=None,
        init_belief=False,
    ):
        """
        Initialize the ProblemSolver.
        
        Args:
            story (str): The narrative text containing the scenario
            question (str): The question to be answered about the story
            choices (list): List of possible answer choices
            K (int): Number of hypotheses to propose for each variable
            assigned_model (list, optional): Predefined variables for manual mode
            model_name (str): Model type ("automated" or specific model name)
            episode_name (str): Unique identifier for this problem instance
            llm (str): Language model for inference tasks
            hypo_llm (str): Language model for hypothesis generation
            verbose (bool): Whether to print detailed progress information
            dataset_name (str): Name of the dataset being processed
            hypo_method (str): Method for hypothesis generation
            nested (bool): Whether to use nested reasoning for higher-order ToM
            tab (str): Indentation string for nested output formatting
            full (bool): Whether to use full context or abbreviated processing
            video_id (str, optional): Video identifier for multimodal datasets
            answerfunc (callable): Function to select answer from probabilities
            back_inference (bool): Whether to use backward inference
            reduce_hypotheses (bool): Whether to reduce observation hypotheses
            precomputed_states (list, optional): Pre-extracted state information
            precomputed_actions (list, optional): Pre-extracted action information
            prev_hyp (list, optional): Previous hypotheses for reuse
            no_model_adjustment (bool): Disable model adjustment for ablation studies
            recursion_depth (int): Depth of nested reasoning recursion
            nested_timeline_before (list, optional): Previous nested timeline data
            nested_time_variables_before (list, optional): Previous nested variables
            init_belief (bool): Whether initial belief state is known
        """
        # Core problem data
        self.story = story
        self.question = question
        self.choices = choices
        self.K = K  # Number of hypotheses to propose for each variable
        
        # Model configuration
        self.assigned_model = deepcopy(assigned_model)
        self.llm = llm
        self.hypo_llm = hypo_llm
        self.model_name = model_name
        self.episode_name = episode_name
        self.verbose = verbose
        self.dataset_name = dataset_name
        self.hypo_method = hypo_method
        
        # Agent and nested reasoning setup
        self.inf_agent_name = "NONE"  # Will be set during parsing
        self.nested_agents_list = []
        self.orig_choices = deepcopy(choices)
        self.orig_story = "NONE"
        self.nested = nested
        self.tab = tab
        self.full = full
        self.video_id = video_id
        self.answerfunc = answerfunc
        
        # Processing state and results
        self.intermediate_node_results = []
        self.back_inference = back_inference
        self.reduce_hypotheses = reduce_hypotheses
        self.prev_hyp = prev_hyp
        self.estimation_dictionary = {}
        self.translate_id_recorder = {}
        
        # Timing and cost tracking
        self.start_time = time.time()
        self.middle_result_time = self.start_time
        self.no_model_adjustment = no_model_adjustment
        self.recursion_depth = recursion_depth
        self.nested_timeline_before = nested_timeline_before
        self.nested_time_variables_before = nested_time_variables_before
        
        # Initialize cost tracking
        self.start_cost = (
            probs.cost_of_estimating_likelihood
            + utils.cost_of_information_extracting
            + utils.cost_of_proposing_hypotheses
        )
        self.middle_result_cost = self.start_cost
        self.start_api = (
            probs.times_of_estimating
            + utils.times_of_information_extracting
            + utils.times_of_proposing_hypotheses
        )
        self.middle_api = self.start_api
        
        # Precomputed data and state flags
        self.states = precomputed_states
        self.actions = precomputed_actions
        self.initial_state = "NONE"
        self.memory = False
        self.realistic = False
        self.ALL_VARIABLES = [
            "State", "Action", "Utterance", "Belief", "Goal", "Observation"
        ]  # All possible variables for extraction
        self.init_belief = init_belief

        # Import helper functions from other modules
        self.clear_current_nodes = NodeResultTracker.clear_current_nodes
        self.translate_and_add_node_results = NodeResultTracker.translate_and_add_node_results
        self.infer_last_timestamp = TimestepInference.infer_last_timestamp
        self.infer_belief_at_timestamp = TimestepInference.infer_belief_at_timestamp
        self.load_parsed_result_into_self = ProblemParser.load_parsed_result_into_self
        self.parse_story_and_question = ProblemParser.parse_story_and_question
        self.contains_utterance = utils.contains_utterance
        self.check_nested = utils.check_nested
        self.get_nested_states = Nested.get_nested_states
        self.save_nested_results = Nested.save_nested_results

    def information_extraction(self):
        """
        Extract temporal mental state variables from the story context.
        
        This is the core information extraction step that processes the story
        to identify and extract temporal variables (states, actions, beliefs, etc.)
        at each timestep. Handles both automated and manual variable selection,
        and manages nested reasoning for higher-order Theory of Mind.

        Returns:
            tuple: Contains:
                - time_variables: List of Variable objects at each timestep
                - variable_values_with_time: Timeline table with extracted values
                - no_observation_hypothesis: Hypothesis for unobserved cases
                - all_timesteps: Total number of timesteps in the story
        """
        context = ""
        
        # Step 1: Determine which variables to extract based on mode
        if self.model_name == "automated":
            var_types = self.ALL_VARIABLES  # Use all possible variables
        else:
            var_types = self.assigned_model  # Use predefined variable set
            
        # Step 2: Handle nested reasoning episode naming
        # For nested states, episodes at same timestep share timeline
        if "nestedChunk" in self.episode_name:
            orig_episode_name = deepcopy(self.episode_name)
            self.episode_name = "_".join(
                self.episode_name.split("_")[:-1]
            )  # Remove "state hypotheses" suffix
            
        # Step 3: Create timeline extractor and load existing data
        timeline = Timeline.TimeLine(
            self.story,
            self.question,
            self.choices,
            var_types,
            self.model_name,
            self.episode_name,
            self.inf_var_name,
            self.llm,
            self.dataset_name,
        )
        
        # Load previously computed timeline if available
        variable_values_with_time = Timeline.load_timeline_table(
            self.model_name, self.episode_name
        )
        # Step 4: Extract timeline data if not already available
        if "MMToM" not in self.dataset_name:
            if variable_values_with_time is None:
                if self.model_name == "automated":
                    # Try to load without reuse first
                    variable_values_with_time = Timeline.load_timeline_table(
                        self.model_name, self.episode_name, reuse=False
                    )
                    if variable_values_with_time is None:
                        # Extract fresh timeline data
                        variable_values_with_time, init_belief = timeline.extract()
                        self.init_belief = init_belief
                    else:
                        # Supply additional extraction to existing data
                        variable_values_with_time, init_belief = (
                            timeline.supply_extraction(variable_values_with_time)
                        )
                        self.init_belief = init_belief
                else:
                    # Manual mode: extract timeline directly
                    variable_values_with_time, init_belief = timeline.extract()
                    self.init_belief = init_belief
                    
                if self.verbose:
                    print(variable_values_with_time)
            else:
                # Handle missing agent action variables in loaded data
                if (
                    (len(variable_values_with_time) == 1) and
                    f"{self.inf_agent_name}'s Action" not in
                    variable_values_with_time[0]
                ):
                    new_variable_values_with_time = deepcopy(
                        variable_values_with_time)
                    for k in variable_values_with_time[0].keys():
                        if "'s Action" in k and self.inf_agent_name in k:
                            new_variable_values_with_time[0][
                                f"{self.inf_agent_name}'s Action"
                            ] = variable_values_with_time[0][k]
                    variable_values_with_time = new_variable_values_with_time

                # Determine initial belief state from action data
                if (
                    len(variable_values_with_time) == 1 and
                    variable_values_with_time[0][
                        f"{self.inf_agent_name}'s Action"] == "NONE"
                ):
                    self.init_belief = True
                else:
                    self.init_belief = False
                    
        # Step 5: Restore episode name for nested reasoning
        if "nestedChunk" in self.episode_name:
            self.episode_name = orig_episode_name
        # Step 6: Handle nested reasoning setup
        if self.nested:
            # Find relevant entities for nested reasoning
            self.relevant_entities = utils.find_relevant_entities(
                self.choices, self.nested_agents_list, self.llm
            )
            
            # Create ground truth timeline for comparison
            ground_truth_timeline = Timeline.TimeLine(
                self.orig_story,
                self.question,
                self.orig_choices,
                self.assigned_model,
                self.model_name,
                f"{self.episode_name}_ground_truth",
                inf_var="Belief",
                llm=self.llm,
                dataset_name=self.dataset_name,
            )
            
            # Load or extract ground truth timeline
            ground_truth_variable_values_with_time = Timeline.load_timeline_table(
                self.model_name, f"{self.episode_name}_ground_truth"
            )
            if ground_truth_variable_values_with_time is None:
                ground_truth_variable_values_with_time, _ = (
                    ground_truth_timeline.extract()
                )
                if self.verbose:
                    print(ground_truth_variable_values_with_time)

        # Step 7: Load or generate time variables for inference
        reuse = False
        if self.nested is True:
            # Enable reuse for nested reasoning across models
            reuse = True

        if self.nested:
            # For nested reasoning, final variables are stored with _depth2 suffix
            time_variables = ElementExtractor.load_time_variables(
                self.model_name, self.episode_name + "_depth2"
            )
        else:
            time_variables = ElementExtractor.load_time_variables(
                self.model_name, self.episode_name, reuse=reuse
            )

        # Step 8: Prepare variable types for extraction
        variable_types = []
        for var_name in self.assigned_model:
            variable_types.append((self.inf_agent_name, var_name))
        variable_types.append(("", "All Actions"))

        if self.model_name == "automated":
            # For automated model, only extract basic states and actions
            # Hypothesis generation happens later in the pipeline
            variable_types = [
                ("", "State"),
                ("", "All Actions"),
                (self.inf_agent_name, "Action"),
            ]

        # Step 9: Generate time variables if not already available
        if time_variables is None:
            time_variables = ElementExtractor.get_variables_with_time(
                variable_values_with_time,
                variable_types,
                self.inf_agent_name,
                self.inf_var_name,
                context,
                self.choices,
                self.K,
                self.llm,
                self.hypo_llm,
                self.verbose,
                self.hypo_method,
                self.dataset_name,
                self.full,
                self.initial_state,
                self.prev_hyp,
                self.states,
                self.actions,
                self.question,
            )

            # Step 10: Handle nested reasoning if enabled
            if self.nested:
                # Nested reasoning workflow:
                # GT States -> Reconstructed but not solved States -> Solved belief of states
                reconstructed_but_not_solved_time_variables = deepcopy(time_variables)
                reconstructed_but_not_solved_variable_values_with_time = deepcopy(
                    variable_values_with_time
                )
                
                # Update episode name with depth information
                self.episode_name += "_depth" + str(len(self.nested_agents_list))
                
                # Load or generate ground truth time variables
                ground_truth_time_variables = ElementExtractor.load_time_variables(
                    self.model_name, f"{self.episode_name}_gt"
                )
                if ground_truth_time_variables is None:
                    print(ground_truth_variable_values_with_time)
                    ground_truth_time_variables = ElementExtractor.get_variables_with_time(
                        ground_truth_variable_values_with_time,
                        variable_types,
                        self.inf_agent_name,
                        self.inf_var_name,
                        context,
                        self.orig_choices,
                        self.K,
                        self.llm,
                        self.hypo_llm,
                        self.verbose,
                        self.hypo_method,
                        self.dataset_name,
                        self.full,
                        self.initial_state,
                    )
                    
                # Save ground truth and reconstructed variables
                ElementExtractor.save_time_variables(
                    ground_truth_time_variables,
                    self.model_name,
                    f"{self.episode_name}_gt",
                )
                ElementExtractor.save_time_variables(
                    reconstructed_but_not_solved_time_variables,
                    self.model_name,
                    f"{self.episode_name}_reconstructed_but_not_solved",
                )
                
                # Load or solve nested states
                solved_time_variables = ElementExtractor.load_time_variables(
                    self.model_name, self.episode_name
                )
                if solved_time_variables is None:
                    solved_time_variables = self.get_nested_states(
                        self,
                        reconstructed_but_not_solved_time_variables,
                        ground_truth_time_variables,
                        reconstructed_but_not_solved_variable_values_with_time,
                        ground_truth_variable_values_with_time,
                    )
                    ElementExtractor.save_time_variables(
                        solved_time_variables, self.model_name, self.episode_name
                    )
                # Step 11: Process nested agents iteratively
                while len(self.nested_agents_list) > 1:
                    self.nested_agents_list = self.nested_agents_list[1:]
                    if len(self.nested_agents_list) == 1:
                        time_variables = deepcopy(solved_time_variables)
                        break
                        
                    # Reconstruct story from current agent's perspective
                    self.first_agent_name = self.nested_agents_list[0]
                    self.story, vis = utils.reconstruct_story_nested(
                        self.story, self.first_agent_name, self.llm, self.dataset_name
                    )
                    self.story = " ".join(self.story)

                    # Update episode name for current depth
                    self.episode_name = self.episode_name[:-1] + str(
                        len(self.nested_agents_list)
                    )
                    ElementExtractor.save_reconstructed_story(
                        vis, self.model_name, self.episode_name, self.first_agent_name
                    )
                    
                    # Extract timeline from reconstructed story
                    reconstructed_but_not_solved_variable_values_with_time, _ = (
                        Timeline.TimeLine(
                            self.story,
                            self.question,
                            self.choices,
                            self.assigned_model,
                            self.model_name,
                            self.episode_name,
                            self.inf_var_name,
                            self.llm,
                            self.dataset_name,
                        ).extract(inferred_agent=self.nested_agents_list[-1])
                    )
                    
                    # Load or solve time variables for current depth
                    solved_time_variables = ElementExtractor.load_time_variables(
                        self.model_name, self.episode_name
                    )
                    if solved_time_variables is None:
                        reconstructed_but_not_solved_time_variables = (
                            ElementExtractor.get_variables_with_time(
                                reconstructed_but_not_solved_variable_values_with_time,
                                variable_types,
                                self.inf_agent_name,
                                self.inf_var_name,
                                context,
                                self.choices,
                                self.K,
                                self.llm,
                                self.hypo_llm,
                                self.verbose,
                                self.hypo_method,
                                self.dataset_name,
                                self.full,
                                self.initial_state,
                                self.prev_hyp,
                            )
                        )
                        solved_time_variables = self.get_nested_states(
                            self,
                            reconstructed_but_not_solved_time_variables,
                            ground_truth_time_variables,
                            reconstructed_but_not_solved_variable_values_with_time,
                            ground_truth_variable_values_with_time,
                        )
                        ElementExtractor.save_time_variables(
                            solved_time_variables, self.model_name, self.episode_name
                        )
                    
                    # Update ground truth for next iteration
                    ground_truth_variable_values_with_time = deepcopy(
                        reconstructed_but_not_solved_variable_values_with_time
                    )
                    ground_truth_time_variables = deepcopy(solved_time_variables)

            # Step 12: Save final time variables
            ElementExtractor.save_time_variables(time_variables, self.model_name, self.episode_name)

        # Step 13: Update timing and cost metrics
        self.middle_result_time = time.time()
        self.middle_result_cost = (
            probs.cost_of_estimating_likelihood
            + utils.cost_of_information_extracting
            + utils.cost_of_proposing_hypotheses
        )
        self.middle_api = (
            probs.times_of_estimating
            + utils.times_of_information_extracting
            + utils.times_of_proposing_hypotheses
        )

        # Step 14: Find no-observation hypothesis for unobserved cases
        no_observation_hypothesis = "NONE"
        for variables in time_variables:
            for key, val in variables.items():
                if "Observation" in key and val.is_observed is False:
                    no_observation_hypothesis = val.possible_values[-1]
                    break

        # Step 15: Calculate total timesteps
        all_timesteps = len(time_variables)

        return (
            time_variables,
            variable_values_with_time,
            no_observation_hypothesis,
            all_timesteps,
        )

    def solve_with_automated_model(
        self,
        time_variables,
        all_timesteps,
        no_observation_hypothesis,
        variable_values_with_time,
        all_probs,
    ):
        """
        Solve the problem using automated model discovery.
        
        This method dynamically discovers which mental state variables to track
        and uses backward inference to solve the problem. It works by starting
        from the final timestep and working backwards, discovering the optimal
        model structure at each step.

        Args:
            time_variables: List of Variable objects at each timestep
            all_timesteps: Total number of timesteps in the story
            no_observation_hypothesis: Hypothesis for unobserved cases
            variable_values_with_time: Timeline table with extracted values
            all_probs: List to store probability results

        Returns:
            tuple: Contains:
                - results: Final probability distribution over answer choices
                - model_record: Record of discovered models at each timestep
        """
        # Step 1: Initialize preprocessing data
        preproposed_ob_hypos = []
        last_state = "None"

        # Extract precomputed states from time variables
        precomputed_states = []
        for tim_var in time_variables:
            if "State" in tim_var:
                precomputed_states.append(tim_var["State"].possible_values[0])
            else:
                precomputed_states.append("NONE")

        # Step 2: Set up agent-specific variable names
        inf_agent_action = "NONE"
        belief_name = f"{self.inf_agent_name}'s Belief"

        # Step 3: Determine if utterances are present and propose initial model
        contain_utterance = self.contains_utterance(
            self, time_variables, variable_values_with_time
        )
        proposed_model = model_adjustment.initial_model_proposal(
            self.story + self.question,
            self.inf_var_name,
            contain_utterance,
        )
        
        # Step 4: Initialize model tracking structures
        assigned_models = {}  # Maps timestep to discovered model
        saved_model_variables = {}  # Maps timestep to saved variables

        # Step 5: Backward inference loop - start from final timestep
        for start_timestep in range(all_timesteps - 1, -1, -1):
            print(f"Starting from timestep {start_timestep}")
            
            # Step 6: Set up previous belief for current timestep
            if start_timestep > 0 and belief_name in time_variables[start_timestep - 1]:
                # Use actual previous belief if available
                previous_belief = deepcopy(
                    time_variables[start_timestep - 1][belief_name]
                )
                previous_belief.name = "Previous Belief"
            else:
                # Initialize with default belief
                previous_belief = ElementExtractor.Variable(
                    "Previous Belief", True, False, ["NONE"], np.ones(1)
                )

            # Step 7: Set up output file for node results
            output_folder = "../results/node_results"
            file_path = f"{output_folder}/automated_{self.episode_name}_back{int(self.back_inference)}_reduce{int(self.reduce_hypotheses)}.csv"

            # Step 8: Assign proposed model to current timestep
            assigned_models[start_timestep] = proposed_model
            
            # Step 9: Run model discovery for current timestep
            results, terminate, assigned_models, saved_model_variables = (
                model_adjustment.model_discovery(
                    start_timestep,
                    all_timesteps,
                    self.verbose,
                    time_variables,
                    previous_belief,
                    self.inf_agent_name,
                    self.inf_var_name,
                    self.estimation_dictionary,
                    self.infer_last_timestamp,
                    no_observation_hypothesis,
                    variable_values_with_time,
                    all_probs,
                    self.answerfunc,
                    argmax,
                    argmin,
                    ElementExtractor.save_belief_probs,
                    self.model_name,
                    self.episode_name,
                    self.infer_belief_at_timestamp,
                    belief_name,
                    ElementExtractor.get_variables_at_time,
                    ElementExtractor.mmtom_get_variables_at_time,
                    self.choices,
                    self.K,
                    self.llm,
                    self.hypo_llm,
                    self.hypo_method,
                    self.full,
                    preproposed_ob_hypos,
                    last_state,
                    inf_agent_action,
                    assigned_models,
                    file_path,
                    self.clear_current_nodes,
                    self.dataset_name,
                    self.states,
                    self.actions,
                    self.question,
                    precomputed_states,
                    saved_model_variables,
                    self.no_model_adjustment,
                    self,
                )
            )
            
            # Step 10: Check if inference can terminate early
            if terminate:
                model_record = {
                    "Initial model propose": proposed_model,
                    "Assigned models": assigned_models,
                }
                print(model_record)
                return results, model_record

        # Step 11: Return final results if no early termination
        model_record = {
            "Initial model propose": proposed_model,
            "Assigned models": assigned_models,
        }
        print(model_record)
        return results, model_record

    def solve(self):
        """
        Main solving method that orchestrates the complete problem-solving pipeline.
        
        Handles both automated and manual modes, determines problem type (realistic,
        memory, nested), and routes to appropriate solving strategy.

        Returns:
            tuple: Contains:
                - final_probs: Probability distribution over answer choices
                - model_record: Record of models used (empty for manual mode)
        """
        # Step 1: Handle trivial case with single choice
        if len(self.choices) == 1:
            return [1.0], {}
            
        # Step 2: Parse and classify the problem
        if self.model_name == "automated":
            # Determine if this is a realistic question (can be answered from state)
            if model_adjustment.determine_realistic_questions(self.question) is True:
                self.realistic = True
                self.nested = False
                
            # Determine if nested reasoning is needed
            if self.nested is None:
                self.nested = model_adjustment.determine_higher_order_belief(
                    self.story + self.question)
                print("nested", self.nested)
                if self.check_nested(self) is False:
                    return None, {}
                    
            # Determine if this is a memory question
            if model_adjustment.determine_memory_questions(self.question) is True:
                self.memory = True

        # Step 3: Load or parse story and question
        parsed_result = ElementExtractor.load_parsed_result(
            self.model_name, self.episode_name)
        if parsed_result is None:
            parsed_result = self.parse_story_and_question(self)
        else:
            self.load_parsed_result_into_self(self, parsed_result)

        # Step 4: Handle memory questions with direct LLM query
        if self.memory is True:
            return utils.get_answer_memory_questions(
                self.story, self.question, self.choices, self.llm
            )
        # Step 5: Extract temporal mental state variables
        (
            time_variables,
            variable_values_with_time,
            no_observation_hypothesis,
            all_timesteps,
        ) = self.information_extraction()
        
        # Step 6: Handle realistic questions with direct state lookup
        if self.realistic:
            return utils.get_answer_from_state(
                time_variables[-1]["State"].possible_values[0], self.choices, self.llm
            )
            
        # Step 7: Initialize belief and probability tracking
        previous_belief = ElementExtractor.Variable(
            "Previous Belief", True, False, ["None"], np.ones(1))
        all_probs = []
        self.estimation_dictionary = ElementExtractor.load_estimation_dict(
            self.dataset_name)
        results = None

        # Step 8: Route to appropriate solving method
        if self.model_name == "automated":
            return self.solve_with_automated_model(
                time_variables,
                all_timesteps,
                no_observation_hypothesis,
                variable_values_with_time,
                all_probs,
            )

        else:
            # Step 9: Manual mode with predefined model - backward inference
            for start_timestep in range(all_timesteps - 1, -1, -1):
                print(f"Starting from timestep {start_timestep}")
                belief_name = f"{self.inf_agent_name}'s Belief"
                
                # Step 10: Set up previous belief for current timestep
                if (
                    start_timestep > 0
                    and belief_name in time_variables[start_timestep - 1]
                ):
                    # Use actual previous belief if available
                    previous_belief = deepcopy(
                        time_variables[start_timestep - 1][belief_name]
                    )
                else:
                    # Initialize with default belief
                    previous_belief = ElementExtractor.Variable(
                        "Previous Belief", True, False, ["NONE"], np.ones(1)
                    )
                    
                # Step 11: Process each timestep from start to end
                for i in range(start_timestep, all_timesteps):
                    if self.verbose:
                        print(f"------------- time stamp {i} -------------")
                    time_variables[i]["Previous Belief"] = previous_belief
                    now_variables = []
                    inf_name = f"{self.inf_agent_name}'s {self.inf_var_name}"

                    if i == all_timesteps - 1:
                        # Step 12: Final timestep - perform inference and check termination
                        results, self.estimation_dictionary, all_probs = (
                            self.infer_last_timestamp(
                                self,
                                time_variables=time_variables,
                                i=i,
                                inf_name=inf_name,
                                inf_var_name=self.inf_var_name,
                                now_variables=now_variables,
                                no_observation_hypothesis=no_observation_hypothesis,
                                variable_values_with_time=variable_values_with_time,
                                all_probs=all_probs,
                                all_prob_estimations=self.estimation_dictionary,
                            )
                        )
                        
                        # Step 13: Check if inference can terminate early
                        results = list(results)
                        terminate = False
                        utility_terminate_threshold = -0.673
                        utility = -entropy(results)
                        if len(results) == 2:
                            if utility > utility_terminate_threshold:
                                terminate = True
                        else:
                            if (
                                self.answerfunc is argmax
                                and utility > utility_terminate_threshold
                            ):
                                terminate = True
                            elif self.answerfunc is argmin and min(results) < 0.2:
                                terminate = True
                                
                        if terminate is True:
                            ElementExtractor.save_belief_probs(
                                all_probs, self.model_name, self.episode_name
                            )
                            return results, {}

                    else:
                        # Step 14: Intermediate timestep - update belief state
                        previous_belief, self.estimation_dictionary, all_probs = (
                            self.infer_belief_at_timestamp(
                                self,
                                time_variables=time_variables,
                                i=i,
                                previous_belief=previous_belief,
                                belief_name=belief_name,
                                variable_values_with_time=variable_values_with_time,
                                all_probs=all_probs,
                                no_observation_hypothesis=no_observation_hypothesis,
                                all_prob_estimations=self.estimation_dictionary,
                            )
                        )

        return results, {}


def main(args):
    """
    Main entry point for running Theory of Mind problem solving on datasets.
    
    Loads a dataset, processes each question through the ProblemSolver,
    and evaluates performance. Supports both automated and manual modes
    with comprehensive metrics tracking.

    Args:
        args: Command line arguments containing:
            - dataset_name (str): Name of the dataset (e.g. MMToM-QA, ToMi)
            - K (int): Number of hypotheses to propose for each variable
            - llm_model (str): Language model for inference and hypothesis generation
            - assigned_model (list): Variables for manual mode Bayesian inference
            - automated (bool): Whether to use automated model discovery
            - back_inference (bool): Whether to use backward inference
            - reduce_hypotheses (bool): Whether to reduce observation hypotheses
            - max_num (int): Maximum number of questions to process
            - nested (bool): Whether to use nested reasoning
            - no_model_adjustment (bool): Disable model adjustment for ablation

    Returns:
        None: Prints performance metrics and saves results to files
    """
    # Step 1: Load the Theory of Mind evaluation dataset
    data = DataLoader.load_full_dataset(args.dataset_name)
    
    # Step 2: Initialize tracking variables
    cnt = 0  # Running count of correct responses
    correct = []  # List tracking correctness of each question
    
    # Step 3: Determine model configuration
    if args.automated:
        model_name = "automated"
    else:
        model_name = utils.get_model_name(args.assigned_model)
    print("model_name:", model_name)
    
    llm = args.llm_model
    hypo_method = "guided"
    print(hypo_method)

    # Step 4: Handle HiToM dataset order specification
    order = 0
    if "HiToM" in args.dataset_name:
        order = int(args.dataset_name.split("order")[1])

    # Step 5: Initialize result tracking
    correct_answer_probs = []
    model_records = {}

    # Step 6: Configure inference parameters
    back_inference = args.back_inference
    print(f"Back inference is {back_inference}")
    reduce_hypotheses = args.reduce_hypotheses
    print(f"Reduce observation hypotheses is {reduce_hypotheses}")
    
    # Step 7: Initialize cost and API tracking
    costs = []
    apis = []
    # Step 8: Process each question in the dataset
    for i, d in enumerate(data):
        if i >= args.max_num:
            break
        print(f"Question {i}")
        
        # Step 9: Parse dataset-specific format
        states, actions, video_id = None, None, None
        if "MuMa" in args.dataset_name:
            story, question, choices, correct_answer, video_id = d
        elif "MMToM" in args.dataset_name:
            story, question, choices, correct_answer, states, actions = d
        else:
            story, question, choices, correct_answer = d
            
        # Step 10: Handle HiToM dataset choice filtering
        if "HiToM" in args.dataset_name:
            # Filter choices to only include those mentioned in the story
            orig_choices = deepcopy(choices)
            choices = []
            for j, c in enumerate(orig_choices):
                if c in story:
                    choices.append(c)

        # Step 11: Determine answer selection function
        answerfunc = argmax
        if "LEAST likely" in question:
            answerfunc = argmin

        # Step 12: Create ProblemSolver instance
        solver = ProblemSolver(
            story=story,
            question=question,
            choices=choices,
            K=args.K,
            assigned_model=args.assigned_model,
            model_name=model_name,
            episode_name=f"{args.dataset_name}_{i}",
            llm=llm,
            verbose=True,
            dataset_name=args.dataset_name,
            hypo_method=hypo_method,
            nested=args.nested,
            video_id=video_id,
            answerfunc=answerfunc,
            back_inference=back_inference,
            reduce_hypotheses=reduce_hypotheses,
            precomputed_states=states,
            precomputed_actions=actions,
            prev_hyp=None,
            no_model_adjustment=args.no_model_adjustment,
            recursion_depth=order,
        )

        # Step 13: Solve the problem
        final_probs, model_record = solver.solve()
        model_records[f"Question {i}"] = model_record

        # Step 14: Calculate final timing and cost metrics
        end_time = time.time()
        end_cost = (
            probs.cost_of_estimating_likelihood
            + utils.cost_of_information_extracting
            + utils.cost_of_proposing_hypotheses
        )
        end_api = (
            probs.times_of_estimating
            + utils.times_of_information_extracting
            + utils.times_of_proposing_hypotheses
        )

        # Step 15: Save estimation dictionary for reuse
        ElementExtractor.save_estimation_dict(args.dataset_name, solver.estimation_dictionary)

        # Step 16: Handle cases where model cannot answer
        if final_probs is None:
            print("The assigned model cannot answer the question.")
            correct.append(0)
            utils.enh_print(f"Incorrect, now {cnt} / {i + 1}, {correct}", "red")
            continue

        # Step 17: Determine selected answer and correct answer index
        answer_idx = answerfunc(final_probs)
        if "HiToM" in args.dataset_name:
            # Handle HiToM dataset answer mapping
            true_answer_word = orig_choices[correct_answer]
            correct_idx = -1
            for j, c in enumerate(choices):
                if c == true_answer_word:
                    correct_idx = j
        else:
            # Standard letter-to-number mapping for other datasets
            correct_idx = utils.letter_to_number_mapping(correct_answer)
            
        # Step 18: Extract probability of correct answer
        correct_answer_prob = final_probs[correct_idx]
        correct_answer_probs.append(correct_answer_prob)
        utils.enh_print(f"Likelihood of the correct answer: {correct_answer_prob:.2f}")

        # Step 19: Compile comprehensive metrics
        metrics = {
            "Likelihood of correct answer": correct_answer_prob,
            "Total time": end_time - solver.start_time,
            "Middle time": solver.middle_result_time - solver.start_time,
            "Inference time": end_time - solver.middle_result_time,
            "Total cost": end_cost - solver.start_cost,
            "Middle cost": solver.middle_result_cost - solver.start_cost,
            "Total API": end_api - solver.start_api,
            "Middle API": solver.middle_api - solver.start_api,
            "probs": final_probs,
            "Correctness": answer_idx == correct_idx,
        }
        
        # Step 20: Save metrics to file
        ElementExtractor.save_metrics(
            metrics,
            solver.model_name,
            solver.episode_name,
            solver.back_inference,
            solver.reduce_hypotheses,
        )
        
        # Step 21: Track costs and update counters
        costs.append(end_cost)
        apis.append(end_api)
        utils.enh_print(f"Costs: {costs}", "red")
        utils.enh_print(f"Apis: {apis}", "red")
        
        # Step 22: Evaluate correctness and update tracking
        if answer_idx == correct_idx:
            cnt += 1
            correct.append(1)
            utils.enh_print(f"Correct, now {cnt} / {i + 1}, {correct}")
        else:
            correct.append(0)
            utils.enh_print(f"Incorrect, now {cnt} / {i + 1}, {correct}", "red")

    # Step 23: Analyze model usage for automated mode
    if args.automated:
        model_counts = {}
        for record in model_records.values():
            print("Record:", record)
            assigned_models = record["Assigned models"]
            for model_list in assigned_models.values():
                model_tuple = tuple(model_list)
                model_counts[model_tuple] = model_counts.get(model_tuple, 0) + 1

        print("Model records:", model_records)
        print("Model counts:", model_counts)

    # Step 24: Print final performance summary
    print("Costs:", costs)
    print("Apis:", apis)
    print("Number of correctness:", correct)


if __name__ == "__main__":
    # Step 1: Set up command line argument parser
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=[
            "MMToM-QA", "ToMi-1st", "ToMi-2nd", "ToMi-memory", "ToMi-reality",
            "BigToM_fatb", "BigToM_fafb", "BigToM_fbtb", "BigToM_fbfb", "BigToM_bbfb",
            "MuMaToM_social_goal", "MuMaToM_belief", "MuMaToM_belief_of_goal",
            "HiToM_len1_tell0_order0", "HiToM_len1_tell0_order1", "HiToM_len1_tell0_order2",
            "HiToM_len1_tell0_order3", "HiToM_len1_tell0_order4", "HiToM_len1_tell1_order0",
            "HiToM_len1_tell1_order1", "HiToM_len1_tell1_order2", "HiToM_len1_tell1_order3",
            "HiToM_len1_tell1_order4",
        ],
    )
    
    # Model configuration arguments
    parser.add_argument(
        "--llm_model",
        choices=["gpt-4o"],
        default="gpt-4o",
        help="Language model for inference and hypothesis generation"
    )
    parser.add_argument(
        "--automated", 
        action="store_true", 
        help="Run in automated model discovery mode"
    )
    parser.add_argument(
        "--assigned_model",
        type=str,
        default='["State", "Observation", "Belief", "Action", "Goal"]',
        help="Variables for manual mode Bayesian inference (when automated is false)"
    )
    
    # Inference configuration arguments
    parser.add_argument(
        "--back_inference",
        type=bool,
        default=True,
        help="Use backward inference for automated mode"
    )
    parser.add_argument(
        "--reduce_hypotheses",
        type=bool,
        default=True,
        help="Reduce observation hypotheses for efficiency"
    )
    parser.add_argument(
        "--nested",
        default=None,
        help="Enable nested reasoning for higher-order ToM (None for auto-detect)"
    )
    
    # Processing configuration arguments
    parser.add_argument("--K", type=int, default=1, help="Number of hypotheses per variable")
    parser.add_argument("--max_num", type=int, default=3, help="Maximum questions to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--no_model_adjustment", 
        action="store_true", 
        help="Disable model adjustment (for ablation studies)"
    )
    
    # Step 2: Parse arguments and run main function
    args = parser.parse_args()
    args.assigned_model = ast.literal_eval(args.assigned_model)  # Convert string to list
    main(args)

    # example of command line run with AutoToM, back inference, reduced hypotheses:
    # python ProbSolver.py --automated --dataset_name "MMToM-QA"

    # example of command line run with AutoToM with model specs:
    # python ProbSolver.py --dataset_name ToMi-1st --assigned_model "['State', 'Observation', 'Belief']"
