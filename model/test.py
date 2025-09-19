# test_response_prediction.py
import os
import random
import numpy as np

# Change to model directory
if 'model' not in os.getcwd():
    os.chdir('model')

from ProbSolver import ProblemSolver, argmax
import ElementExtractor

def test_p_v_given_x():
    """Test if AutoToM can find the best graph explaining a given response."""
    
    # Use the playground example
    story = """1 Ella entered the master_bedroom.
    2 Ava entered the master_bedroom.
    3 The onion is in the envelope.
    4 The envelope is in the master_bedroom.
    5 Ella exited the master_bedroom.
    6 Ava moved the onion to the box.
    7 The box is in the master_bedroom.
    8 Ava exited the master_bedroom.
    9 Ella entered the hallway."""

    question = "Where will Ella look for the onion?"
    choices = ["envelope", "box"]
    
    # Step 1: Randomly select a response
    target_v = 'envelope'
    target_response = "box"
    
    # Step 3: Test the current system first
    print("Testing Current System (p(V|X))")
    solver_current = ProblemSolver(
        story=story,
        question=question,
        choices=choices,
        K=1,
        assigned_model=[],
        model_name="automated",
        episode_name="test_current",
        llm="gpt-4o",
        verbose=False,
        dataset_name="test",
        hypo_method="guided",
        nested=False,
        video_id=None,
        answerfunc=argmax,
        back_inference=True,
        reduce_hypotheses=True,
        precomputed_states=None,
        precomputed_actions=None,
        prev_hyp=None,
        no_model_adjustment=False,
        recursion_depth=None
    )
    
    final_probs_current, model_record_current = solver_current.solve()
    print(f"Current system probabilities:\n\tChoices: {choices}\n\tProbabilities: {final_probs_current}")
    return target_v, final_probs_current

if __name__ == "__main__":
    true_response, current_probs = test_p_v_given_x()
    print(f"\nTrue response: {true_response}")
    print(f"Current system prediction: {current_probs}")