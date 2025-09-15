# test_response_prediction.py
import os
import random
import numpy as np

# Change to model directory
if 'model' not in os.getcwd():
    os.chdir('model')

from ProbSolver import ProblemSolver, argmax
import ElementExtractor

def test_response_prediction():
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
    true_response = random.choice(choices)
    true_response = "box"
    print(f"Selected response: {true_response}")
    
    # Step 2: Create a modified story that includes the response as an "observed" action
    # We'll treat the response as if Ella actually said/acted this way
    modified_story = story + f"\n10 Ella said: 'I will look in the {true_response}.'"
    
    print(f"Modified story:\n{modified_story}")
    
    # Step 3: Test the current system first
    print("\n=== Testing Current System (p(q|X)) ===")
    solver_current = ProblemSolver(
        story=story,
        question=question,
        choices=choices,
        K=1,
        assigned_model=[],
        model_name="automated",
        episode_name="test_current",
        llm="gpt-4o",
        verbose=True,
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
    print(f"Current system probabilities: {choices} -> {final_probs_current}")
    
    return true_response, final_probs_current

if __name__ == "__main__":
    true_response, current_probs = test_response_prediction()
    print(f"\nTrue response: {true_response}")
    print(f"Current system prediction: {current_probs}")