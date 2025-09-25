"""Entry point for evaluating models on ToM evals."""

import argparse
import os
import sys
import math
import random

import dotenv
import pandas as pd
import ast
import json
from termcolor import cprint
from openai import OpenAI
import time
from tqdm import tqdm

import ProbSolver
import DataLoader
from ProbSolver import argmax, argmin
import utils

random.seed(888)
sys.path.append(os.path.join('..'))
dotenv.load_dotenv()

EVAL_PATH = os.path.join('..', 'benchmarks', 'full_data_formatted')
REQUIRED_COLUMNS = ['story', 'question', 'answer_choices', 'gt_answer']
DATA_PATH = os.path.join('..', 'results', 'exp1')


def get_logits(
    info: str,
    question: str,
    choices: list[str],
    model: str = "gpt-4o",
    ) -> list[float]:
    """
    Get logprobs from base LM aligned with answer_choices indexing.
    Returns normalized posterior probabilities where probabilities_list[i] corresponds to choices[i].
    """
    global accumulated_cost_logits
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    inst = f"""
    Answer the question based on the story.
    Story: {info}
    Question: {question}
    """
    letter_choices = utils.return_letters(len(choices))
    format_choices = ""
    for i, c in enumerate(choices):
        format_choices += f"{letter_choices[i]}) {c}\n"
    prompt = f"{inst}{format_choices}Answer:"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            seed=42,
        )
        resp_json = response.model_dump_json(indent=2)
        resp_dict = json.loads(resp_json)
        logprobs = resp_dict["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        # Extract probabilities for each choice letter
        probabilities = [0.0] * len(choices)
        for logprob_item in logprobs:
            token = logprob_item["token"]
            if token in letter_choices:
                idx = letter_choices.index(token)
                probabilities[idx] = math.exp(logprob_item["logprob"])
        # Check if we got all choices
        # TODO: Need to handle when LM does not provide logprobs for all choices
        # Currently, this causes P(V | R, X) to fail since it needs an index
        missing_choices = [letter_choices[i] for i, p in enumerate(probabilities) if p == 0.0]
        if missing_choices:
            print(f"Warning: Missing logprobs for choices: {missing_choices}")
            print(f"Available logprobs: {[item['token'] for item in logprobs]}")
        # Normalize to proper posteriors
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback: uniform distribution if no valid probabilities
            probabilities = [1.0 / len(choices)] * len(choices)
        return probabilities
    except Exception as e:
        print(f"retrying due to an error {e}")
        time.sleep(20)
        return get_logits(info, question, choices, model)


def letter_to_number_mapping(letter):
    return ord(letter.upper()) - ord("A")


def load_eval(eval_name):
    """Load the eval from the EVAL_PATH."""
    ## MMToM-QA
    if eval_name.lower() == "mmtom-qa":
        eval_path = os.path.join(EVAL_PATH, eval_name)
    ## ToMi
    elif eval_name.lower() == "tomi-first":
        eval_path = os.path.join(EVAL_PATH, 'tomi_first_order.csv')
    elif eval_name.lower() == "tomi-second":
        eval_path = os.path.join(EVAL_PATH, 'tomi_second_order.csv')
    elif eval_name.lower() == "tomi-memory":
        eval_path = os.path.join(EVAL_PATH, 'tomi_memory.csv')
    elif eval_name.lower() == "tomi-reality":
        eval_path = os.path.join(EVAL_PATH, 'tomi_reality.csv')
    return pd.read_csv(eval_path)


def main(args):
    """Main entry point for evaluating models on ToM evals."""
    # Grab the eval data
    eval_data = DataLoader.load_full_dataset(args.eval_name)
    # Ensure the eval data has the required columns
    assert all(col in eval_data.columns for col in REQUIRED_COLUMNS)
    print(f"Eval data shape: {eval_data.shape}")
    # Metric tracking
    # Dictionary of Q:argmax_V P(V|X) pairs for each question Q (the MAP)
    vs_map_estimates = []
    # Running list tracking where response is correct for Q
    questions_correct = []
    # Evaluation metrics
    eval_metrics = []
    # Iterate through the eval data (each row is a question Q)
    for index, row in tqdm(
        list(eval_data.iterrows())[:50], desc="Evaluating"):
        story = row['story']
        question = row['question']
        answer_choices = row['answer_choices']
        gt_answer = row['gt_answer']
        cprint(f"Story: {story}", "green")
        cprint(f"Question: {question}", "red")
        cprint(f"choices: {answer_choices}", "blue")
        answerfunc = argmin if "LEAST likely" in question else argmax
        # Compute P(R | X) = f(X) from LMs
        lm_response_posterior = get_logits(
            story, question, answer_choices, args.llm)
        # lm_response_posterior = [0.9, 0.1]
        # NOTE: Remove this after testing
        # lm_response_posterior = [0.9, 0.1]
        cprint(f"LM response posterior: {lm_response_posterior}", "yellow")
        lm_answer_idx = answerfunc(lm_response_posterior)
        correct_idx = letter_to_number_mapping(gt_answer)
        questions_correct.append(int(lm_answer_idx == correct_idx))
        eval_metrics.append({
            "correct": lm_answer_idx == correct_idx,
            "posterior": lm_response_posterior,
            "answer_choices": answer_choices,
            "gt_answer": gt_answer,
            "response": answer_choices[lm_answer_idx],
            "question_index": index,
            "model": args.llm
        })
        # Compute P(R | X) \propto P(V, X) from AutoTom
        if args.fit_to_responses:
            compute_response = True
            observed_response_idx = lm_answer_idx
            autotom_model_name = "autotom_fit_to_response"
        else:
            compute_response = False
            observed_response_idx = None
            autotom_model_name = "autotom"
        p_v_x = ProbSolver.ProblemSolver(
            story=story,
            question=question,
            choices=answer_choices,
            K=args.K,
            model_graph=args.model_graph,
            model_name="automated",
            episode_name=f"{args.eval_name}_{index}",
            llm=args.llm,
            verbose=args.verbose,
            eval_name=args.eval_name,
            hypo_method="guided",
            nested=args.nested,
            video_id=None,
            answerfunc=answerfunc,
            back_inference=args.back_inference,
            reduce_hypotheses=args.reduce_hypotheses,
            precomputed_states=None,
            precomputed_actions=None,
            prev_hyp=None,
            no_model_adjustment=False,
            recursion_depth=None,
            compute_response=compute_response,
            observed_response_idx=observed_response_idx,
            observed_response_probs=lm_response_posterior
        )
        response_posterior, v_map = p_v_x.solve()
        cprint(f"AutoTom response posterior: {response_posterior}", "yellow")
        vs_map_estimates.append({
            "question_index": index,
            "v_map": v_map,
        })
        if response_posterior is None:
            questions_correct.append(0)
            continue
        answer_idx = answerfunc(response_posterior)
        correct_idx = letter_to_number_mapping(gt_answer)
        questions_correct.append(int(answer_idx == correct_idx))
        eval_metrics.append({
            "correct": answer_idx == correct_idx,
            "posterior": response_posterior,
            "answer_choices": answer_choices,
            "gt_answer": gt_answer,
            "response": answer_choices[answer_idx],
            "question_index": index,
            "model": autotom_model_name
        })
    # Store the metrics
    eval_metrics = pd.DataFrame(eval_metrics)
    vs_map_estimates = pd.DataFrame(vs_map_estimates)
    if args.fit_to_responses:
        eval_metrics.to_csv(
            f"{DATA_PATH}/eval_metrics_{args.eval_name}_fit_to_responses.csv", index=False)
        vs_map_estimates.to_csv(
            f"{DATA_PATH}/vs_map_estimates_{args.eval_name}_fit_to_responses.csv", index=False)
    else:
        eval_metrics.to_csv(
            f"{DATA_PATH}/eval_metrics_{args.eval_name}.csv", index=False)
        vs_map_estimates.to_csv(
            f"{DATA_PATH}/vs_map_estimates_{args.eval_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_name',
        type=str,
        required=True,
        choices=[
            "mmtom-qa",
            "tomi-first",
            "tomi-second",
            "tomi-memory",
            "tomi-reality",
            "bigtom-fatb",
            "bigtom-fafb",
            "bigtom-fbtb",
            "bigtom-fbfb",
            "bigtom-bbfb",
            "mumatom-social-goal",
            "mumatom-belief",
            "mumatom-belief-of-goal",
            "hitom-len1-tell0-order0",
            "hitom-len1-tell0-order1",
            "hitom-len1-tell0-order2",
            "hitom-len1-tell0-order3",
            "hitom-len1-tell0-order4",
            "hitom-len1-tell1-order0",
            "hitom-len1-tell1-order1",
            "hitom-len1-tell1-order2",
            "hitom-len1-tell1-order3",
            "hitom-len1-tell1-order4",
        ],
    )
    parser.add_argument(
        "--llm",
        choices=["gpt-4o"],
        default="gpt-4o",
        help="Language model for inference and hypothesis generation"
    )
    parser.add_argument(
        "--fit_to_responses",
        action="store_true",
        help="Fit to responses"
    )
    parser.add_argument(
        "--back_inference",
        type=bool,
        default=True,
        help="Use backward inference for automated mode"
    )
    parser.add_argument(
        "--model_graph",
        type=str,
        default='["State", "Observation", "Belief", "Action", "Goal"]',
        help="Variables for manual mode Bayesian inference (when automated is false)"
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
    parser.add_argument(
        "--K", type=int, default=5, help="Number of hypotheses per variable")
    parser.add_argument(
        "--max_num", type=int, default=3, help="Maximum questions to process")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    args.model_graph = ast.literal_eval(args.model_graph)
    main(args)
