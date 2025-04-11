import os
import json
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json, score_answer
from string import Template
from tqdm.auto import tqdm
import argparse

MAX_TURNS = 4  # Maximum number of conversation turns
MAX_MESSAGES_PER_CHAR = 4
RPBENCH_PATH = "data/rpbench_character.jsonl"

# Define the 6 criteria for evaluation
CRITERIA = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

TEMPLATE = Template(
    """$background

# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text
"""
)

CRITERIA_EVALUATION_TEMPLATE = Template(
    """You are an expert in roleplay analysis. Given the following roleplay interaction, evaluate the assistant's response based on six criteria:
$criteria

NPC Profile:
## Name: $name_text
## Title: $title
## Description: $description

User input: $user_input
Assistant response: $response

Please return a JSON object with scores (from 1.0 to 10.0) for:
$criteria
"""
)

def evaluate_response(model, npc_profile, response,user_input):
    """Evaluate a single response using the 6 criteria"""
    prompt = CRITERIA_EVALUATION_TEMPLATE.substitute(
        name_text=npc_profile["name_text"],
        title=npc_profile["title"],
        description=npc_profile["description"],
        user_input=user_input,
        response=response,
        criteria=", ".join(CRITERIA),
    )
    
    # messages = [{"role": "user", "content": prompt}]
    evaluation = score_answer(prompt)
    
    try:
        # scores = extract_and_parse_json(evaluation)
        scores = evaluation
        # Convert all scores to float and calculate total
        total_score = sum(float(scores.get(key, 0)) for key in CRITERIA)
        return total_score, scores
    except:
        # If JSON parsing fails, return 0 score
        return 0, {}

def compare_models(model_1, model_2):
    model_1_total = 0
    model_2_total = 0
    model_1_wins = 0
    model_2_wins = 0
    model_1_turns = 0
    model_2_turns = 0
    comparison_results = []
    
    # Track all metrics
    model_1_metrics = {key: 0 for key in CRITERIA}
    model_2_metrics = {key: 0 for key in CRITERIA}
    
    # Load evaluation data
    eval_data = []
    with jsonlines.open(RPBENCH_PATH) as reader:
        for obj in reader:
            eval_data.append(obj)
    # eval_data = eval_data[:10]
    print(f"Loaded {len(eval_data)} examples from {RPBENCH_PATH}")

    # Load models
    candidate_config = make_config("config/api_config.yaml")
    assert model_1 in candidate_config, f"{model_1} not found in candidate config"
    assert model_2 in candidate_config, f"{model_2} not found in candidate config"
    print(f"Comparing `{model_1}` and `{model_2}`")

    # Evaluation model for scoring responses
    evaluator_config = make_config("config/judger_config.yaml")
    evaluator_model = list(evaluator_config.values())[0]
    
    for d in (pbar := tqdm(eval_data)):
        npc_profile = d["npc_profile"]
        background = d["background"]
        greeting = "\n".join(d["conversation"][0]["sentences"])
        # Prepare system prompt for candidates
        system_prompt = TEMPLATE.substitute(background=background, **npc_profile, user_input=greeting)
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting}
        ]
        
        current_speaker = model_1  # Start with model_1
        for turn in range(MAX_TURNS):
            # calc n turns
            if current_speaker == model_1:
                model_1_turns += 1
            elif current_speaker == model_2:
                model_2_turns += 1
            # Generate user input 
            
            # Get response from current speaker
            current_response = chat_completion(candidate_config[current_speaker], messages.copy())
            
            # Evaluate response
            current_score, current_scores = evaluate_response(
                evaluator_model, npc_profile, current_response, messages[-1]["content"]
            )
            
            # get last message
            last_message = messages[-1]
            # Track results
            result = {
                "npc_profile": npc_profile,
                "turn": turn + 1,
                "speaker": current_speaker,
                "user_input": last_message["content"],
                "response": current_response,
                "scores": current_scores,
                "total_score": current_score
            }
            
            # Update metrics
            if current_speaker == model_1:
                model_1_total += current_score
                for key in model_1_metrics:
                    model_1_metrics[key] += current_scores.get(key, 0)
            else:
                model_2_total += current_score
                for key in model_2_metrics:
                    model_2_metrics[key] += current_scores.get(key, 0)
            
            comparison_results.append(result)
            
            # Update progress bar
            pbar.set_postfix({
                f"{model_1}_score": model_1_total,
                f"{model_2}_score": model_2_total,
                f"{model_1}_wins": model_1_wins,
                f"{model_2}_wins": model_2_wins,
                "current_turn": turn + 1
            })
            # replace all user with assistant and assistant with user in current messages
            for message in messages:
                if message["role"] == "user":
                    message["role"] = "assistant"
                elif message["role"] == "assistant":
                    message["role"] = "user"
            # Continue conversation with the current response
            messages.append({"role": "user", "content": current_response})
            
            # Switch speaker for next turn
            current_speaker = model_2 if current_speaker == model_1 else model_1

        # Determine winner for this conversation divide by number of turns
        model_1_total = round(model_1_total, 3)
        model_2_total = round(model_2_total, 3)
        if model_1_total > model_2_total:
            model_1_wins += 1
        elif model_2_total > model_1_total:
            model_2_wins += 1

    # Save results
    if not os.path.exists("results/character"):
        os.makedirs("results/character")
    # devide keys by turns for metrics
    for key in model_1_metrics:
        model_1_metrics[key] /= model_1_turns
        model_1_metrics[key] = round(model_1_metrics[key], 2)
    for key in model_2_metrics:
        model_2_metrics[key] /= model_2_turns
        model_2_metrics[key] = round(model_2_metrics[key], 2)
    result_summary = {
        "model_1": model_1,
        "model_2": model_2,
        "model_1_total_score": model_1_total,
        "model_2_total_score": model_2_total,
        "model_1_avg_total_score": model_1_total / model_1_turns,
        "model_2_avg_total_score": model_2_total / model_2_turns,
        "model_1_turns": model_1_turns,
        "model_2_turns": model_2_turns,
        "model_1_wins": model_1_wins,
        "model_2_wins": model_2_wins,
        "model_1_metrics": model_1_metrics,
        "model_2_metrics": model_2_metrics,
        "conversations": comparison_results
    }
    
    with open(f"results/character/comparison_{model_1}_vs_{model_2}.json", "w") as f:
        json.dump(result_summary, f, indent=2)
        
    # Print summary
    print("\nFinal Results:")
    print(f"{model_1} total score: {model_1_total}")
    print(f"{model_2} total score: {model_2_total}")
    print(f"{model_1} wins: {model_1_wins}")
    print(f"{model_2} wins: {model_2_wins}")
    print("\nDetailed Metrics:")
    print(f"{model_1} metrics:")
    for metric, value in model_1_metrics.items():
        print(f"  {metric}: {value}")
    print(f"\n{model_2} metrics:")
    for metric, value in model_2_metrics.items():
        print(f"  {metric}: {value}")
    
    return result_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, default="gpt-4")
    args = parser.parse_args()
    compare_models(args.model_1, args.model_2)