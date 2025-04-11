import os
import argparse
import yaml
import json
import pandas as pd
from glob import glob
import tqdm

def load_model_mappings(config_path):
    """Load model name to beautiful name mappings from config file"""
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)
    return {k: v["beautiful_name"] for k, v in model_config.items()}

def load_results(result_dir, subset, model_filter=None):
    """Load all result files for a given subset"""
    result_files = glob(os.path.join(result_dir, subset, "benchmark_*_1runs.json"))
    results = []
    
    for file in tqdm.tqdm(result_files, desc="Loading results", unit="file"):
        with open(file, "r") as f:
            data = json.load(f)

            # fix zero results by multiplying the avg score by len of runs: reuslts list, then count the zeros then divide by the count minus number of zeros
            n_zeros = 0
            for i in range(len(data["runs"][0]["results"])):
                if data["runs"][0]["results"][i]["total_score"] == 0:
                    n_zeros += 1

            if n_zeros > 0:
                # update the final_avg_score by multiplying by the len of results. then divide by the count minus number of zeros
                len_of_results = len(data["runs"][0]["results"])
                print(f"Found {n_zeros} zero results for model {data['model']}. Adjusting final_avg_score from {data['final_avg_score']}")
                data["final_avg_score"] = data["final_avg_score"] * len_of_results / (len_of_results - n_zeros)
                print(f"Updated final_avg_score: {data['final_avg_score']}")

            results.append({
                "model": data["model"],
                # "num_runs": data["num_runs"],
                "final_avg_score": data["final_avg_score"],
                "metrics": data["final_metrics"]
            })
        
    # Filter results if model_filter is provided
    if model_filter:
        results = [res for res in results if res["model"] in model_filter]
    # Sort by final_avg_score
    # results.sort(key=lambda x: x["final_avg_score"], reverse=True)

    return results

def create_leaderboard(results, model_mapping):
    """Create leaderboard from raw results"""
    # Create dataframe from results
    df = pd.DataFrame(results)
    
    # Add beautiful names
    df["beautiful_name"] = df["model"].map(model_mapping)
    
    # Calculate average metrics (assuming metrics is a dict with numeric values)
    # You can customize this part based on which metrics you want to include
    metric_cols = {}
    if len(df) > 0 and "metrics" in df.columns:
        first_metrics = df.iloc[0]["metrics"]
        for metric in first_metrics.keys():
            df[f"metric_{metric}"] = df["metrics"].apply(lambda x: x.get(metric, 0))
            metric_cols[f"metric_{metric}"] = metric
    
    # Sort by final_avg_score (primary) and any other metrics you prefer
    sort_columns = ["final_avg_score"] + list(metric_cols.keys())
    df = df.sort_values(sort_columns, ascending=False)
    
    # Add rank column
    df.insert(0, "rank", range(1, len(df) + 1))
    
    return df, metric_cols

def format_display_df(df, metric_cols):
    """Format the dataframe for display purposes"""
    display_df = df.copy()
    display_df["final_avg_score"] = display_df["final_avg_score"].round(2)
    
    # Rename columns for display
    column_mapping = {
        "beautiful_name": "Model",
        "final_avg_score": "Average Score",
        "num_runs": "Runs"
    }
    
    # Add metric columns to mapping
    for col, name in metric_cols.items():
        column_mapping[col] = name
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Select and order columns
    display_columns = ["rank", "Model", "Average Score"] + list(metric_cols.values()) + ["Runs"]
    return display_df[[c for c in display_columns if c in display_df.columns]]

def create_percentage_df(df, metric_cols):
    """Create a dataframe with values as percentages of the top score"""
    percent_df = df.copy()
    
    # Get numeric columns to convert to percentages
    numeric_cols = ["final_avg_score"] + list(metric_cols.keys())
    
    # Convert each numeric column to percentages
    for col in numeric_cols:
        if col in percent_df.columns:
            max_val = percent_df[col].max()
            if max_val != 0:  # Avoid division by zero
                percent_df[col] = (percent_df[col] / max_val * 100).round(2)
    
    # Rename columns to indicate they're percentages
    column_mapping = {
        "beautiful_name": "Model",
        "final_avg_score": "Average Score",
        "num_runs": "Runs"
    }
    
    # Add metric columns to mapping with percentage indicator
    for col, name in metric_cols.items():
        column_mapping[col] = name
    
    percent_df = percent_df.rename(columns=column_mapping)
    
    # Select and order columns
    display_columns = ["rank", "Model", "Average Score"] + [name for name in metric_cols.values()] + ["Runs"]
    return percent_df[[c for c in display_columns if c in percent_df.columns]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpbench_only", action="store_true",
                        help="Limit to RPBench models only")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--subset", type=str, default="character", 
                       help="Subdirectory to analyze (e.g., 'character' or 'scene')")
    parser.add_argument("--model_config", type=str, default="./config/api_config.yaml")
    args = parser.parse_args()

    rp_bench_models = [
        "claude-3-5-sonnet",
        "claude-3-opus",
        "llama-3-1-405b-instruct-fp8",
        "gpt-4-1106-preview",
        "llama-3.1-70b-instruct",
        "gpt-4o",
        "yi-large",
        "qwen-2-72b-instruct",
        "llama-3-70b-instruct",
        "llama3.1-8B-instruct",
        "mistral-large-2402",
    ]


    # Load model mappings
    model_mapping = load_model_mappings(args.model_config)

    # Load results
    results = load_results(args.result_dir, args.subset,
                           model_filter=rp_bench_models if args.rpbench_only else None)
    
    # Create leaderboard
    leaderboard_df, metric_cols = create_leaderboard(results, model_mapping)
    display_df = format_display_df(leaderboard_df, metric_cols)
    percent_df = create_percentage_df(leaderboard_df, metric_cols)
    
    # Save outputs
    os.makedirs(args.result_dir, exist_ok=True)
    leaderboard_df.to_csv(os.path.join(args.result_dir, f"leaderboard_{args.subset}_full.csv"), index=False)
    display_df.to_csv(os.path.join(args.result_dir, f"leaderboard_{args.subset}_display.csv"), index=False)
    percent_df.to_csv(os.path.join(args.result_dir, f"leaderboard_{args.subset}_percent.csv"), index=False)
    
    # Print results
    print(f"\n{args.subset.capitalize()} Leaderboard:")
    print(display_df.head())
    print(f"\n{args.subset.capitalize()} Percentage Leaderboard:")
    print(percent_df.head())

if __name__ == "__main__":

    main()