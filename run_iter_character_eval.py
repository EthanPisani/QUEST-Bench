import os
import json
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json, score_answer
from string import Template
from tqdm.auto import tqdm
import argparse
import multiprocessing
from functools import partial

MAX_REQUESTS = 4  # Number of predefined user requests
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

# Predefined user requests to modify the first reply
USER_REQUESTS = [
    "Can you elaborate more on that?",
    "Make the character more aggressive/passive.",
    "The character should sound more confident/hesitant.",
    "Make the NPC react more emotionally.",
]

def evaluate_response(model, npc_profile, response, user_input):
    """Evaluate a single response using the 6 criteria"""
    prompt = CRITERIA_EVALUATION_TEMPLATE.substitute(
        name_text=npc_profile["name_text"],
        title=npc_profile["title"],
        description=npc_profile["description"],
        user_input=user_input,
        response=response,
        criteria=", ".join(CRITERIA),
    )
    
    evaluation = score_answer(prompt)
    
    try:
        scores = evaluation
        total_score = sum(float(scores.get(key, 0)) for key in CRITERIA)
        return total_score, scores
    except:
        return 0, {}

def benchmark_single_model(model, num_runs=1, limit=None):
    """Benchmark a single model on the RPBENCH dataset"""
    total_scores = []
    all_results = []
    metrics_accumulator = {key: 0 for key in CRITERIA}
    
    # Load evaluation data
    eval_data = []
    with jsonlines.open(RPBENCH_PATH) as reader:
        for obj in reader:
            eval_data.append(obj)
    print(f"Loaded {len(eval_data)} examples from {RPBENCH_PATH}")
    if limit:
        eval_data = eval_data[:limit]
    # Load model config
    candidate_config = make_config("config/api_config.yaml")
    assert model in candidate_config, f"{model} not found in candidate config"
    print(f"Benchmarking `{model}` with {num_runs} runs")

    # Evaluation model for scoring responses
    evaluator_config = make_config("config/judger_config.yaml")
    evaluator_model = list(evaluator_config.values())[0]
    
    for run in range(num_runs):
        run_scores = []
        run_results = []
        
        for d in (pbar := tqdm(eval_data, desc=f"Run {run+1}/{num_runs}")):
            npc_profile = d["npc_profile"]
            background = d["background"]
            greeting = "\n".join(d["conversation"][0]["sentences"])
            
            # Prepare system prompt
            system_prompt = TEMPLATE.substitute(
                background=background, 
                **npc_profile 
            )
            
            # Initialize conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": greeting}
            ]
            
            # Get initial response
            response = chat_completion(candidate_config[model], messages.copy())
            
            # Evaluate initial response
            # score, scores = evaluate_response(
            #     evaluator_model, npc_profile, response, greeting
            # )
            score, scores = 0, {key: 0 for key in CRITERIA}
            
            # Store initial response results
            initial_result = {
                "npc_profile": npc_profile,
                "request_type": "initial",
                "user_input": greeting,
                "response": response,
                "scores": scores,
                "total_score": score
            }
            run_results.append(initial_result)
            run_scores.append(score)
            
            # Process predefined user requests
            for i, request in enumerate(USER_REQUESTS):
                # Update conversation with previous response and new request
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": greeting},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": request}
                ]
                
                # Get response to the request
                response = chat_completion(candidate_config[model], conversation.copy())
                
                # Evaluate the response
                # score, scores = evaluate_response(
                #     evaluator_model, npc_profile, response, request
                # )
                score, scores = 0, {key: 0 for key in CRITERIA}

                # Store request response results
                request_result = {
                    "npc_profile": npc_profile,
                    "request_type": f"request_{i+1}",
                    "user_input": request,
                    "response": response,
                    "scores": scores,
                    "total_score": score
                }
                run_results.append(request_result)
                run_scores.append(score)
                
                pbar.set_postfix({
                    "avg_score": sum(run_scores)/len(run_scores),
                    "current_request": i+1
                })
        
        # Calculate metrics for this run
        run_metrics = {key: 0 for key in CRITERIA}
        for result in run_results:
            for key in CRITERIA:
                if key in result["scores"]:
                    run_metrics[key] += result["scores"][key]
        
        # Normalize metrics
        for key in run_metrics:
            run_metrics[key] = round(run_metrics[key] / len(run_results), 2)
        
        # Accumulate results across runs
        total_scores.append(sum(run_scores)/len(run_scores))
        all_results.append({
            "run": run+1,
            "average_score": sum(run_scores)/len(run_scores),
            "metrics": run_metrics,
            "results": run_results
        })
        
        # Accumulate metrics for final average
        for key in metrics_accumulator:
            metrics_accumulator[key] += run_metrics[key]
    
    # Calculate final averages
    final_avg_score = sum(total_scores)/len(total_scores)
    for key in metrics_accumulator:
        metrics_accumulator[key] = round(metrics_accumulator[key]/num_runs, 2)
    
    # Prepare final results
    final_results = {
        "model": model,
        "num_runs": num_runs,
        "final_avg_score": final_avg_score,
        "final_metrics": metrics_accumulator,
        "runs": all_results
    }
    
    # Save results
    if not os.path.exists("results/character"):
        os.makedirs("results/character")
    
    with open(f"results/character/benchmark_{model}_{num_runs}runs.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\nFinal Results:")
    print(f"Model: {model}")
    print(f"Number of runs: {num_runs}")
    print(f"Final average score: {final_avg_score:.2f}")
    print("\nDetailed Metrics:")
    for metric, value in metrics_accumulator.items():
        print(f"  {metric}: {value}")
    
    return final_results

def process_model(model, num_runs, limit):
    """Wrapper function for multiprocessing"""
    try:
        print(f"Starting evaluation for model: {model}")
        benchmark_single_model(model, num_runs, limit)
        print(f"Completed evaluation for model: {model}")
    except Exception as e:
        print(f"Error processing model {model}: {str(e)}")

def benchmark_models(model_list, num_runs=1, limit=None):
    """Benchmark multiple models using multiprocessing"""
    num_processes = min(multiprocessing.cpu_count(), len(model_list))
    print(f"Running benchmark for {len(model_list)} models using {num_processes} processes")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(partial(process_model, num_runs=num_runs, limit=limit), model_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path to .txt file containing model names (one per line)")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate")
    args = parser.parse_args()
    
    if args.model.endswith('.txt'):
        # Batch mode - read models from file
        with open(args.model, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
        benchmark_models(models, args.num_runs, args.limit)
    else:
        # Single model mode
        benchmark_single_model(args.model, args.num_runs, args.limit)