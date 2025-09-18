# test_response_prediction.py
import os
import random
import numpy as np

# Change to model directory
if 'model' not in os.getcwd():
    os.chdir('model')

from ProbSolver import ProblemSolver, argmax
import ElementExtractor
from BayesianInference import BayesianInferenceModel
import model_adjustment


parent_graph = {
        "Observation": ["State"],
        "Belief": ["Previous Belief", "Observation"],
        "Action": ["Goal", "Belief", "Emotion", "Belief of Goal"],
        "Utterance": ["Goal", "Belief", "Emotion"],
        "Emotion": ["Goal", "Belief"],
        "Expression": ["Emotion"],
        "Response": ["Goal", "Belief", "Utterance"],
    }

def visualize_agent_model(assigned_models):
    """Visualize agent model as ASCII graph in terminal."""
    
    print("\n" + "="*50)
    print("AGENT MODEL VISUALIZATION")
    print("="*50)
    
    for timestep, model_vars in assigned_models.items():
        print(f"\nTimestep {timestep}:")
        print(f"Variables: {model_vars}")
        
        # Build graph connections
        connections = []
        for var in model_vars:
            if var in parent_graph:
                parents = [p for p in parent_graph[var] if p in model_vars]
                if parents:
                    connections.append(f"{' → '.join(parents)} → {var}")
                else:
                    connections.append(f"{var} (root)")
        
        # Print graph
        for conn in connections:
            print(f"  {conn}")
    
    print("="*50)

def compute_p_r_given_v_x_using_original_code(
    story: str,
    question: str,
    choices: list,
    observed_response: str,
    agent_model: list) -> tuple[float, dict]:
    """
    Compute P(R|V,X) using the original code infrastructure.
    
    Args:
        story: The ToM story
        question: The question being asked
        choices: Available answer choices
        observed_response: The observed response we want to explain
        agent_model: The specific agent model V to test (should include Response)
        
    Returns:
        tuple: (probability of observed response, model details)
    """
    print(f"\n--- Computing P(R='{observed_response}'|V={agent_model},X) using original code ---")
    
    try:
        # Create a solver with this specific model using manual mode
        solver = ProblemSolver(
            story=story,
            question=question,
            choices=choices,
            K=1,
            assigned_model=agent_model,
            model_name="manual",  # Use manual mode with specific model
            episode_name=f"test_r_given_v_{hash(tuple(agent_model))}",
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
            no_model_adjustment=True,  # Skip model adjustment for manual mode
            recursion_depth=None
        )
        
        # Solve to get P(R|V,X) for this model
        final_probs, model_record = solver.solve()
        
        if final_probs is not None:
            # Get the probability of the observed response
            response_idx = choices.index(observed_response)
            response_prob = final_probs[response_idx]
            
            print(f"P(R='{observed_response}'|V={agent_model},X) = {response_prob:.4f}")
            
            return response_prob, {
                'probabilities': final_probs,
                'response_prob': response_prob,
                'model_record': model_record
            }
        else:
            print(f"Failed to solve for model {agent_model}")
            return 0.0, {
                'probabilities': None,
                'response_prob': 0.0,
                'model_record': None
            }
            
    except Exception as e:
        print(f"Error testing model {agent_model}: {e}")
        return 0.0, {
            'probabilities': None,
            'response_prob': 0.0,
            'model_record': None
        }

def compute_p_v_given_r_x_using_original_code(
    story: str,
    question: str,
    choices: list,
    observed_response: str) -> tuple[list, dict]:
    """
    Compute P(V|R,X) using the original code infrastructure and MODEL_SPACE.
    
    Uses Bayes' rule: P(V|R,X) = P(R|V,X) * P(V) / P(R|X)
    Since P(R|X) is constant for all models, we use: P(V|R,X) ∝ P(R|V,X) * P(V)
    
    Args:
        story: The ToM story
        question: The question being asked
        choices: Available answer choices
        observed_response: The observed response we want to explain
        
    Returns:
        tuple: (probabilities for each model, model details)
    """
    # Use the original MODEL_SPACE from model_adjustment.py
    # Filter to only include models that have Response variable for P(R|V,X)
    model_space_with_response = [model for model in model_adjustment.MODEL_SPACE if 'Response' in model]
    
    print(f"\n=== Computing P(V|R,X) for observed response: '{observed_response}' ===")
    print(f"Using original MODEL_SPACE with Response variable")
    print(f"Testing {len(model_space_with_response)} different agent models...")
    
    model_likelihoods = []
    model_details = {}
    
    for i, model_vars in enumerate(model_space_with_response):
        print(f"\n--- Testing Agent Model {i+1}: {model_vars} ---")
        
        # For P(V|R,X), we compute P(R|V,X) for each model V
        response_prob, details = compute_p_r_given_v_x_using_original_code(
            story=story,
            question=question,
            choices=choices,
            observed_response=observed_response,
            agent_model=model_vars
        )
        
        model_likelihoods.append(response_prob)
        model_details[tuple(model_vars)] = {
            'agent_model': model_vars,
            'p_r_given_v_x': response_prob,
            'details': details
        }
    
    # Convert likelihoods to probabilities using Bayes' rule
    # P(V|R,X) ∝ P(R|V,X) * P(V)
    # Assuming uniform prior P(V), so P(V|R,X) ∝ P(R|V,X)
    
    if sum(model_likelihoods) > 0:
        # Normalize to get probabilities
        model_probs = np.array(model_likelihoods) / sum(model_likelihoods)
    else:
        # If all models failed, use uniform distribution
        model_probs = np.ones(len(model_space_with_response)) / len(model_space_with_response)
    
    print(f"\n=== P(V|R,X) Results ===")
    for i, (model_vars, prob) in enumerate(zip(model_space_with_response, model_probs)):
        print(f"P(V={model_vars}|R='{observed_response}',X) = {prob:.4f}")
    
    return model_probs.tolist(), model_details

def test_response_prediction():
    """Test both P(R|V,X) and P(V|R,X) functionality using original code."""
    
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
    
    # Step 1: Randomly select a response (this is what we're trying to explain)
    observed_response = random.choice(choices)
    print(f"Observed response: {observed_response}")
    
    # Step 2: Test the current system first (p(q|X))
    print("\n=== Testing Current System (p(q|X)) ===")
    try:
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
        print("P(q|X) results:", final_probs_current)
        print("Model record:", model_record_current)
    except Exception as e:
        print(f"Error in current system: {e}")
        print("Skipping current system test due to API key issue...")
    
    # Step 3: Test P(R|V,X) - individual model testing using original code
    print("\n" + "="*60)
    print("TESTING P(R|V,X) - RESPONSE PROBABILITY GIVEN AGENT MODEL")
    print("="*60)
    
    # Test a specific agent model from the original MODEL_SPACE
    test_agent_model = ['State', 'Observation', 'Belief', 'Action', 'Goal', 'Response']
    response_prob, details = compute_p_r_given_v_x_using_original_code(
        story=story,
        question=question,
        choices=choices,
        observed_response=observed_response,
        agent_model=test_agent_model
    )
    
    print(f"\nP(R='{observed_response}'|V={test_agent_model},X) = {response_prob:.4f}")
    
    # Step 4: Test P(V|R,X) - agent model inference from response using original code
    print("\n" + "="*60)
    print("TESTING P(V|R,X) - AGENT MODEL INFERENCE FROM RESPONSE")
    print("="*60)
    
    model_probs, model_details = compute_p_v_given_r_x_using_original_code(
        story=story,
        question=question,
        choices=choices,
        observed_response=observed_response
    )
    
    # Find the most likely model
    best_model_idx = np.argmax(model_probs)
    best_model = list(model_details.keys())[best_model_idx]
    best_prob = model_probs[best_model_idx]
    
    print(f"\n=== Best Agent Model ===")
    print(f"Most likely model: {list(best_model)}")
    print(f"Probability: {best_prob:.4f}")
    
    # Visualize the best model
    visualize_agent_model({0: list(best_model)})
    
    # Step 5: Show detailed comparison
    print(f"\n=== Detailed Model Comparison ===")
    for model_vars, prob in zip([list(k) for k in model_details.keys()], model_probs):
        details = model_details[tuple(model_vars)]
        print(f"Model {model_vars}:")
        print(f"  P(V|R,X) = {prob:.4f}")
        print(f"  P(R|V,X) = {details['p_r_given_v_x']:.4f}")
        print()

if __name__ == "__main__":
    test_response_prediction()