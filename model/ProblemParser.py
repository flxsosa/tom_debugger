import copy

import utils
import ElementExtractor as EE


def load_parsed_result_into_self(self, parsed_result):
    needed_list = [
        "story", "question", "choices", "inf_agent_name", "first_agent_name",
        "inf_var_name", "orig_choices", "orig_story", "initial_state", "nested_agents_list",
        "full"
    ]
    for key, value in parsed_result.items():
        if key in needed_list:
            setattr(self, key, value)


def parse_story_and_question(self):
    """
    Extracts and processes information from the story, question and choices.
    Sets up necessary instance variables and saves the extracted information.
    
    Returns:
        info: Dictionary containing the extracted and processed information,
                or None if no model is assigned
    """
    if self.model_graph is None:
        print("There is no model graph defined")
        return None
    # Extract info from question
    question_info = EE.get_info_from_question(
        self.question, self.llm, self.eval_name)
    # MuMA-specific handling for fusing video and text story
    if "muma" in self.eval_name:
        video_story = EE.video_extracted_actions(self.video_id)
        if "belief_of_goal" in self.eval_name:
            video_story = utils.correct_visual_actions(
                video_story, self.choices, self.llm)
        self.story = utils.story_fusion(video_story, self.story, self.llm)
        if self.eval_name in ["mumatom-belief", "mumatom-social-goal"]:
            self.initial_state = EE.get_initial_state(self.story, self.llm)
            self.story = EE.find_inference_timestep(
                self.story, self.choices, self.llm)
            if question_info != "":
                question_info = utils.rewrite_belief_info(
                    question_info, self.initial_state, self.llm)
    # Edit the story text to have no prefixes and contain the question info
    self.story = utils.remove_story_prefixes(self.story)
    self.story += " " + question_info
    # HiToM-specific handling for story text
    if "hitom" in self.eval_name:
        self.story = utils.split_story_sentences(self.story, self.llm)
    # HiToM-specific handling for nested reasoning
    if self.nested:
        self.orig_story = copy.deepcopy(self.story)
        self.orig_choices = utils.rephrase_choices(
            self.question, self.orig_choices, self.hypo_llm)
        self.nested_agents_list = EE.find_nested_agent_list(
            self.question, self.choices, self.hypo_llm)
        self.first_agent_name = self.nested_agents_list[0]
        self.story, vis = utils.reconstruct_story_nested(
            self.orig_story, self.first_agent_name, self.llm, self.eval_name
        )
        self.story = " ".join(self.story)
        EE.save_reconstructed_story(
            vis, self.model_name, self.episode_name, self.first_agent_name)
        self.question = utils.rephrase_question_nested(
            self.question,
            self.nested_agents_list[-1],
            self.llm,
            self.eval_name
        )
    # Rephrase the choices into full sentences for LMs
    self.choices = utils.rephrase_choices(
        self.question, self.choices, self.hypo_llm)
    # Determine the variable to infer for the given question
    # If we're inferring the model grah (automated mode), infer the variable
    if self.model_name == "automated":
        possible_vars = ["Belief", "Goal"]
        # Eval-specific possible variables (eval-specific variable priors)
        # NOTE: We might want to remove these priors later for P(r|V,X)
        if "bigtom" in self.eval_name:
            possible_vars.append("Action")
        if "tomi" in self.eval_name or "hitom" in self.eval_name:
            possible_vars = ["Belief"]
        self.inf_var_name = EE.get_inf_var(
            self.question,
            self.choices,
            possible_vars,
            self.hypo_llm,
            self.eval_name,
        )
    # Otherwise, grab the variable from the model graph
    else:
        self.inf_var_name = EE.get_inf_var(
            self.question,
            self.choices,
            self.model_graph,
            self.hypo_llm,
            self.eval_name,
        )
    # Find the agent to infer the variable for
    # If nested, it's the last dude in the chain of dudes
    if self.nested:
        self.inf_agent_name = self.nested_agents_list[-1]
    # Otherwise, infer the agent that the question is about
    else:
        self.inf_agent_name = utils.find_inferred_agent(
            self.question, self.choices, self.hypo_llm
        )
    # Make sure that we have all the necessary info to solve the problem
    necessary_info_kinds = [
        "story",
        "question",
        "choices",
        "inf_agent_name",
        "first_agent_name",
        "inf_var_name",
        "orig_choices",
        "orig_story",
        "initial_state",
        "nested_agents_list",
        "full"
    ]
    # Aggregate the necessary info into a dictionary to be used for inference
    info = {k:v for k,v in self.__dict__.items() if k in necessary_info_kinds}
    # Save the necessary info to a file
    EE.save_parsed_result(info, self.model_name, self.episode_name)
    return info

