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
RPBENCH_PATH = "data/rpbench_scene.jsonl"

# Define the criteria for scene evaluation
CRITERIA = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

TEMPLATE = Template(
    """# Background:    
$background

# NPC Profile:
$npc_profile

# Previous plots recap:
$plot_recap

# Current Objectives:
$objectives

# When to end the scene:
$progression

# NPC Status:
$npc_status

You are an AI NPC for a role-play game. Based on the previous information, you will play a NPC character in the scene.
"""
)

CRITERIA_EVALUATION_TEMPLATE = Template(
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

# Predefined user requests to modify the scene interaction
USER_REQUESTS = [
    "Can we explore a different approach to this situation?",
    "Let's focus more on the emotional aspects of this scene.",
    "The NPC should be more assertive in this situation.",
    "Can we wrap up this scene and move to the next plot point?",
]

def evaluate_response(model, scene_context, response, user_input):
    """Evaluate a single scene response using the criteria"""
    prompt = CRITERIA_EVALUATION_TEMPLATE.substitute(
        background=scene_context["background"],
        objectives=scene_context["objectives"],
        progression=scene_context["progression"],
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
    """Benchmark a single model on the scene evaluation dataset"""
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
        skip_n = 0
        # check for already existing results
        if os.path.exists(f"results/scene/temp_benchmark_{model}_{num_runs}runs.json"):
            # load existing results into run_results
            with open(f"results/scene/temp_benchmark_{model}_{num_runs}runs.json", "r") as f:
                run_results = json.load(f)["runs"][0]["results"]
                skip_n = len(run_results) // (len(USER_REQUESTS) +1)
                # skip_n = skip_n - 1 # starting message
                print(f"Loaded existing results for run {run+1} for model {model}")
                # get run_scores from existing results
                for result in run_results:
                    run_scores.append(result["total_score"])
                print(f"Loaded {len(run_results)} existing results for run {run+1} for model {model}")
                print(f"Skipping {skip_n} examples for run {run+1} for model {model}")

        for d in (pbar := tqdm(eval_data[skip_n: ], desc=f"Run {run+1}/{num_runs}")):
            scene_context = {
                "background": d["background"],
                "objectives": d["objectives"],
                "progression": d["progression"],
                "npc_profile": d["npc_profile"],
                "npc_status": d["npc_status"],
                "plot_recap": d["plot_recap"]
            }
            
            greeting = "\n".join(d["conversation"][0]["sentences"])
            
            # Prepare system prompt
            system_prompt = TEMPLATE.substitute(**d)
            
            # Initialize conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": greeting}
            ]
            
            # Get initial response
            response = chat_completion(candidate_config[model], messages.copy())
            
            # Evaluate initial response
            # score, scores = evaluate_response(
            #     evaluator_model, scene_context, response, greeting
            # )
            score, scores = 0, {key: 0 for key in CRITERIA}
            # Store initial response results
            initial_result = {
                "scene_context": scene_context,
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
                #     evaluator_model, scene_context, response, request
                # )
                score, scores = 0, {key: 0 for key in CRITERIA}
                
                # Store request response results
                request_result = {
                    "scene_context": scene_context,
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
                # save intermediate results
                with open(f"results/scene/temp1_benchmark_{model}_{num_runs}runs.json", "w") as f:
                    json.dump({
                        "model": model,
                        "num_runs": num_runs,
                        "runs": [{
                            "run": run+1,
                            "results": run_results
                        }]
                    }, f, indent=2)
        
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
    if not os.path.exists("results/scene"):
        os.makedirs("results/scene")
    
    with open(f"results/scene/benchmark_{model}_{num_runs}runs.json", "w") as f:
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