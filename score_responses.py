import os
import json
import jsonlines
from glob import glob
from tqdm import tqdm
import argparse
from utils import score_answer
from string import Template

# Same criteria as original evaluation
CRITERIA = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]
CRITERIA_EVALUATION_TEMPLATE_SCENE = Template(
    """You are an expert in roleplay scene analysis. Given the following scene interaction, evaluate the assistant's response based on six criteria:
$criteria

Scene Context:
## Background: $background
## Objectives: $objectives
## Progression Points: $progression

User input: $user_input
Assistant response: $response

Please return a JSON object with scores (from 1.0 to 10.0) for:
$criteria
"""
)

CRITERIA_EVALUATION_TEMPLATE_CHARACTER = Template(
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

def evaluate_response(profile, response, user_input, roleplay_type="character"):
    """Evaluate a single response using the 6 criteria (same as original)"""
    prompt = ""
    if roleplay_type == "scene":
        prompt = CRITERIA_EVALUATION_TEMPLATE_SCENE.substitute(
            background=profile["background"],
            objectives=profile["objectives"],
            progression=profile["progression"],
            user_input=user_input,
            response=response,
            criteria=", ".join(CRITERIA),
        )
    elif roleplay_type == "character":
        prompt = CRITERIA_EVALUATION_TEMPLATE_CHARACTER.substitute(
            name_text=profile["name_text"],
            title=profile["title"],
            description=profile["description"],
            user_input=user_input,
            response=response,
            criteria=", ".join(CRITERIA),
        )
        
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            evaluation = score_answer(prompt)
            scores = evaluation
            total_score = sum(float(scores.get(key, 0)) for key in CRITERIA)
            return total_score, scores
        except Exception as e:
            if attempt == max_attempts - 1:  # Last attempt failed
                print(f"Evaluation failed after {max_attempts} attempts. Error: {e}")
                return 0, {}
            continue
    
    return 0, {}

def process_file(input_file, output_dir,roleplay_type="character"):
    """Process a single JSON file and evaluate all responses"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    model_name = data["model"]
    num_runs = data["num_runs"]
    
    evaluated_data = {
        "model": model_name,
        "num_runs": num_runs,
        "final_avg_score": 0,
        "final_metrics": {key: 0 for key in CRITERIA},
        "runs": []
    }
    
    for run in data["runs"]:
        run_results = []
        run_metrics = {key: 0 for key in CRITERIA}
        total_score = 0
        
        for result in run["results"]:
            # Evaluate each response
            if roleplay_type == "scene":
                profile = result["scene_context"]
            elif roleplay_type == "character":
                profile = result["npc_profile"]
            user_input = result["user_input"]
            response = result["response"]
            
            score, scores = evaluate_response(profile, response, user_input, roleplay_type)
            
            evaluated_result = {
                "request_type": result["request_type"],
                "user_input": user_input,
                "response": response,
                "scores": scores,
                "total_score": score
            }
            if roleplay_type == "scene":
                evaluated_result["scene_context"] = profile
            elif roleplay_type == "character":
                evaluated_result["npc_profile"] = profile
            
            run_results.append(evaluated_result)
            total_score += score
            for key in CRITERIA:
                run_metrics[key] += scores.get(key, 0) 
        
        # Calculate averages for this run
        num_responses = len(run_results)
        run_avg_score = total_score / num_responses if num_responses > 0 else 0
        for key in run_metrics:
            run_metrics[key] = round(run_metrics[key] / num_responses, 2)
        
        evaluated_data["runs"].append({
            "run": run["run"],
            "average_score": run_avg_score,
            "metrics": run_metrics,
            "results": run_results
        })
    
    # Calculate final averages across all runs
    total_runs = len(evaluated_data["runs"])
    if total_runs > 0:
        evaluated_data["final_avg_score"] = sum(
            run["average_score"] for run in evaluated_data["runs"]
        ) / total_runs
        
        for key in CRITERIA:
            evaluated_data["final_metrics"][key] = round(
                sum(run["metrics"][key] for run in evaluated_data["runs"]) / total_runs,
                2
            )
    
    # Save evaluated file
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_file, 'w') as f:
        json.dump(evaluated_data, f, indent=2)
    print(f"Evaluated file saved to {output_file}")
    
    return evaluated_data

def evaluate_responses(input_dir, output_dir, roleplay_type="character"):
    """Evaluate all responses in JSON files from input_dir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = sorted(glob(os.path.join(input_dir, "benchmark_*.json")))
    # check if files exist in output_dir
    out_files = sorted(glob(os.path.join(output_dir, "benchmark_*.json")))
    # get difference between files and out_files
    files = [f for f in files if os.path.basename(f) not in [os.path.basename(out_file) for out_file in out_files]]
    if len(files) == 0:
        print("All files have already been evaluated. Exiting.")
        return

    bar = tqdm(total=len(files), desc="Evaluating files")
    for file in files:
        bar.set_postfix_str(file.split("/")[-1])
        bar.refresh()
        process_file(file, output_dir, roleplay_type)
        bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate responses using the same criteria as original evaluation")
    parser.add_argument("--input_dir", type=str, default="responses/character",
                      help="Directory containing JSON files with responses")
    parser.add_argument("--output_dir", type=str, default="scores/character",
                      help="Directory to save evaluated results")
    parser.add_argument("--roleplay_type", type=str, choices=["character", "scene"], default="character",
                        help="Type of roleplay to evaluate (character or scene)")
    
    args = parser.parse_args()
    
    print(f"Evaluating responses in {args.input_dir}")
    print(f"Saving results to {args.output_dir}")
    
    evaluate_responses(args.input_dir, args.output_dir, args.roleplay_type)
    print("Evaluation completed!")