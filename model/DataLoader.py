import csv
import random
import pandas as pd


def load_full_dataset(dataset_name):
    random.seed(42)
    dataset_name = dataset_name.lower()

    if "hitom-len1" in dataset_name:
        order = dataset_name.split('order')[1]
        tell = dataset_name.split('tell')[1][0]
        if tell == "1":
            tell = "tell"
        else:
            tell = "no_tell"
        with open(f"../benchmarks/full_data_formatted/hitom-len1_{tell}_{order}.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = eval(choices)
            # choices[1] = choices[1][1:]
            random.shuffle(choices)
            # print(choices[0], true_answer)
            answer = 0
            for i, c in enumerate(choices):
                if c == true_answer:
                    answer = i
            # print(choices, answer)
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "mumatom-belief":
        with open("../benchmarks/full_data_formatted/MuMA_belief.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        return data

    elif dataset_name == "mumatom-social-goal":
        with open("../benchmarks/full_data_formatted/MuMA_social_goal.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        # print(data)
        return data

    elif dataset_name == "mumatom-belief_of_goal":
        with open(
            "../benchmarks/full_data_formatted/MuMA_belief_of_goal.csv", "r"
        ) as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        # print(data)
        return data

    if dataset_name == "mmtom-qa":
        with open("../benchmarks/full_data_formatted/mmtom-qa.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices, answer = eval(row[2]), row[3]
            states, actions = eval(row[4]), eval(row[5])

            random.shuffle(choices)
            
            if choices[0] == answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer, states, actions))
        return data

    if dataset_name == "tomi-first":
        with open("../benchmarks/full_data_formatted/tomi_first_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return pd.DataFrame(
            data, columns=['story', 'question', 'answer_choices', 'gt_answer'])
    
    if dataset_name == "tomi-second":
        with open("../benchmarks/full_data_formatted/tomi_second_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return pd.DataFrame(
            data, columns=['story', 'question', 'answer_choices', 'gt_answer'])
    
    if dataset_name == "tomi-memory":
        with open("../benchmarks/full_data_formatted/tomi_memory.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return pd.DataFrame(
            data, columns=['story', 'question', 'answer_choices', 'gt_answer'])
    
    if dataset_name == "tomi-reality":
        with open("../benchmarks/full_data_formatted/tomi_reality.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return pd.DataFrame(
            data, columns=['story', 'question', 'answer_choices', 'gt_answer'])

    elif "fantom" in dataset_name:
        if dataset_name == "fantom-first-tb-short":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:accessible_short_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-first-tb-full":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:accessible_full_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-first-fb-short":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:inaccessible_short_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-first-fb-full":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:inaccessible_full_context_first-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-second-tb-short":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:accessible_short_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-second-tb-full":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:accessible_full_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-second-fb-short":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:inaccessible_short_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "fantom-second-fb-full":
            with open(
                "../benchmarks/full_data_formatted/fantom_tom:belief:inaccessible_full_context_second-order.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[3], row[2]

            choices = eval(choices)
            random.shuffle(choices)
            if choices[0] == true_answer:
                answer = "A"
            elif choices[1] == true_answer:
                answer = "B"
            else:
                print("ERROR")
                
            data.append((story, question, choices, answer))
        return data

    elif "bigtom" in dataset_name:
        if dataset_name == "bigtom-fbfb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_forward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom-bbfb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_backward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_bbtb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_backward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_fafb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_forward_action_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_fatb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_forward_action_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom-fbfb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_forward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_fbtb":
            with open(
                "../benchmarks/full_data_formatted/bigtom_forward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[3], row[2]
            if dataset_name == "bigtom":
                choices = choices.replace("[", "").replace("]", "").split(",")
            else:
                choices = choices.replace("[", "").replace("]", "").split(";")
                
            cleaned_choices = []
            for c in choices:
                cleaned_choices.append(c.strip())
            choices = cleaned_choices

            random.shuffle(choices)
            if choices[0].replace(".", "") == true_answer.replace(".", ""):
                answer = "A"
            elif choices[1].replace(".", "") == true_answer.replace(".", ""):
                answer = "B"
            else:
                print("ERROR")

            data.append((story, question, choices, answer))
        return data


def load_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    random.seed(42)
    if "bigtom" in dataset_name:
        if dataset_name == "bigtom":
            with open("../benchmarks/mini-data/bigtom.csv", "r") as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom-bbfb":
            with open(
                "../benchmarks/mini-data/bigtom_backward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_bbtb":
            with open(
                "../benchmarks/mini-data/bigtom_backward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_fafb":
            with open(
                "../benchmarks/mini-data/bigtom_forward_action_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_fatb":
            with open(
                "../benchmarks/mini-data/bigtom_forward_action_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom-fbfb":
            with open(
                "../benchmarks/mini-data/bigtom_forward_belief_false_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        elif dataset_name == "bigtom_fbtb":
            with open(
                "../benchmarks/mini-data/bigtom_forward_belief_true_belief_stories.csv",
                "r",
            ) as f:
                reader = csv.reader(f, delimiter=",")
                rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            if dataset_name == "bigtom":
                choices = choices.replace("[", "").replace("]", "").split(",")
            else:
                choices = choices.replace("[", "").replace("]", "").split(";")
                
            cleaned_choices = []
            for c in choices:
                cleaned_choices.append(c.strip())
            choices = cleaned_choices

            random.shuffle(choices)
            print(choices)
            if choices[0].replace(".", "") == true_answer.replace(".", ""):
                answer = "A"
            elif choices[1].replace(".", "") == true_answer.replace(".", ""):
                answer = "B"
            else:
                print("ERROR")

            data.append((story, question, choices, answer))
        return data

    if dataset_name == "tomi-first":
        with open("../benchmarks/mini-data/tomi_first_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            choices[1] = choices[1][1:]
            random.shuffle(choices)
            # print(choices[0], true_answer)
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
            # print(choices, answer)
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "fantom-first":
        with open("../benchmarks/mini-data/fantom_first.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[3], row[2]

            choices = eval(choices)
            random.shuffle(choices)
            if choices[0] == true_answer:
                answer = "A"
            elif choices[1] == true_answer:
                answer = "B"
            else:
                print("ERROR")
                
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "mumatom-belief":
        with open("../benchmarks/mini-data/MuMa_belief.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
        return data

    elif dataset_name == "mumatom-social-goal":
        with open("../benchmarks/mini-data/MuMa_social_goal.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
            
        return data

    elif dataset_name == "mumatom-belief-of-goal":
        with open("../benchmarks/mini-data/MuMa_belief_of_goal.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            choices = eval(row[2])
            new_choices = []
            for c in choices:
                c = c.replace("A) ", "")
                c = c.replace("B) ", "")
                c = c.replace("C) ", "")
                new_choices.append(c)
            answer = row[3][0]
            id = row[4]
            data.append((story, question, new_choices, answer, id))
            
        return data

    if dataset_name == "tomi-second":
        with open("../benchmarks/mini-data/tomi_second_order.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []
        for row in rows[1:]:
            story, question = row[0], row[1]
            true_answer, choices = row[2], row[3]
            choices = (
                choices.replace("[", "").replace("]", "").replace("'", "").split(",")
            )
            choices[1] = choices[1][1:]
            random.shuffle(choices)
            
            if choices[0] == true_answer:
                answer = "A"
            else:
                answer = "B"
                
            data.append((story, question, choices, answer))
        return data

    if dataset_name == "mmtom-qa":
        with open("../benchmarks/mini-data/mmtom-qa.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            rows = list(reader)
        data = []

        for row in rows[1:]:
            story_and_question, answer = row[0], row[1]
            story = story_and_question.split("Question: ")[0].strip()
            question = (
                story_and_question.split("Question: ")[1].split(" (a)")[0].strip()
            )
            choices = [
                story_and_question.split("(a) ")[1].split(" (b)")[0],
                story_and_question.split("(b) ")[1].split(" Please")[0],
            ]
            answer = answer.capitalize()
            data.append((story, question, choices, answer))
        return data
