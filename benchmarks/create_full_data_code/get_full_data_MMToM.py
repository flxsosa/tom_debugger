import json
import pandas as pd
import random

data_path = "../data/MMToM-QA/multimodal_representations.json"

try:
    data = []
    with open(data_path, "r") as file:
        for line in file:
            d = json.loads(line)
            data.append(d)
    print("JSON file has been successfully loaded.")
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
except FileNotFoundError:
    print(f"No file found at {data_path}")
except Exception as e:
    print(f"An error occurred: {e}")

records = []
for d in data:
    timesteps = d["end_time"] + 1

    actions = d["actions"]
    states = []
    for i in range(timesteps + 1):
        states.append(d[f"state_{i}"])
    
    text = d["question"]
    story = text.split('\nQuestion: ')[0].strip()
    question = text.split('\nQuestion: ')[1].split(" (a) ")[0]
    choices = [
        text.split('\nQuestion: ')[1].split(" (a) ")[1].split(" (b) ")[0].strip(),
        text.split('\nQuestion: ')[1].split(" (a) ")[1].split(" (b) ")[1].split(" Please")[0].strip()
    ]
    if d["answer"] == "a":
        gt_answer = choices[0]
    else:
        gt_answer = choices[1]

    agent_name = actions[0].split(' ')[0]
    new_actions = []
    for action in actions:
        if agent_name not in action:
            action = f"{agent_name} {action}"
        new_actions.append(action)

    record = {
        "story": story,
        "question": question,
        "answer_choices": choices,
        "gt_answer": gt_answer,
        "states": f"{states}",
        "actions": f"{new_actions}"
    }

    assert len(actions) == len(states)
    records.append(record)

output_path_base = "../full_data_formatted/"
df = pd.DataFrame(records)
df.to_csv(f"{output_path_base}MMToM-QA.csv", index=False)

print("Data has been saved to respective files.")
