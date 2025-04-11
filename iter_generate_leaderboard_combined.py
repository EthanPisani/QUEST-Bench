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
    
    for file in tqdm.tqdm(result_files, desc=f"Loading {subset} results", unit="file"):
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
                "subset": subset,
                "final_avg_score": data["final_avg_score"],
                "metrics": data["final_metrics"]
            })
        
    # Filter results if model_filter is provided
    if model_filter:
        results = [res for res in results if res["model"] in model_filter]
    
    return results

def create_leaderboard(results, model_mapping, subset=None):
    """Create leaderboard from raw results"""
    # Create dataframe from results
    df = pd.DataFrame(results)
    
    # Filter by subset if specified
    if subset:
        df = df[df["subset"] == subset]
    
    # Add beautiful names
    df["beautiful_name"] = df["model"].map(model_mapping)
    
    # Calculate average metrics (assuming metrics is a dict with numeric values)
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
def format_display_df2(df, metric_cols):
    """Format the dataframe for display purposes"""
    display_df = df.copy()
    
    # Round all numeric columns
    for col in display_df.select_dtypes(include=['float64']).columns:
        display_df[col] = display_df[col].round(2)
    
    # Rename columns for display
    column_mapping = {
        "beautiful_name": "Model",
        "final_avg_score": "Average Score",
        "final_avg_score_character": "Character Score",
        "final_avg_score_scene": "Scene Score",
        "final_avg_score_combined": "Average Score",
        "num_runs": "Runs"
    }
    
    # Add metric columns to mapping
    for col, name in metric_cols.items():
        column_mapping[col] = name
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Select and order columns
    display_columns = ["rank", "model", "Average Score", "Character Score", "Scene Score"] + list(metric_cols.values()) + ["Runs"]
    return display_df[[c for c in display_columns if c in display_df.columns]]

def format_display_df(df, metric_cols):
    """Format the dataframe for display purposes"""
    display_df = df.copy()
    
    # Round all numeric columns
    for col in display_df.select_dtypes(include=['float64']).columns:
        display_df[col] = display_df[col].round(2)
    
    # Rename columns for display
    column_mapping = {
        "beautiful_name": "Model",
        "final_avg_score": "Average Score",
        "final_avg_score_character": "Character Score",
        "final_avg_score_scene": "Scene Score",
        "final_avg_score_combined": "Average Score",
        "num_runs": "Runs"
    }
    
    # Add metric columns to mapping
    for col, name in metric_cols.items():
        column_mapping[col] = name
    
    display_df = display_df.rename(columns=column_mapping)
    
    # Select and order columns
    display_columns = ["rank", "Model", "Average Score", "Character Score", "Scene Score"] + list(metric_cols.values()) + ["Runs"]
    return display_df[[c for c in display_columns if c in display_df.columns]]

def create_percentage_df(df, metric_cols):
    """Create a dataframe with values as percentages of the top score"""
    percent_df = df.copy()
    
    # Get numeric columns to convert to percentages
    numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['rank', 'num_runs']]
    
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
        "final_avg_score_character": "Character Score",
        "final_avg_score_scene": "Scene Score",
        "final_avg_score_combined": "Average Score",
        "num_runs": "Runs"
    }
    
    # Add metric columns to mapping with percentage indicator
    for col, name in metric_cols.items():
        column_mapping[col] = name
    
    percent_df = percent_df.rename(columns=column_mapping)
    
    # Select and order columns
    display_columns = ["rank", "Model", "Average Score", "Character Score", "Scene Score"] + [name for name in metric_cols.values()] + ["Runs"]
    return percent_df[[c for c in display_columns if c in percent_df.columns]]

def create_combined_leaderboard(character_df, scene_df, model_mapping):
    """Create a combined leaderboard with averaged scores and metrics"""
    # Merge the two dataframes on model
    combined_df = pd.merge(
        character_df[["model", "final_avg_score"] + [c for c in character_df.columns if c.startswith("metric_")]],
        scene_df[["model", "final_avg_score"] + [c for c in scene_df.columns if c.startswith("metric_")]],
        on="model",
        suffixes=("_character", "_scene")
    )
    # print diff on model col with the character and scene dataframes
    diff_models = set(character_df["model"]) - set(scene_df["model"])
    if len(diff_models) > 0:
        print(f"Models in character but not in scene: {diff_models}")
        with open("diff_models.txt", "w") as f:
            for model in diff_models:
                f.write(f"{model}\n")
    # print diff on model col with the scene and character dataframes
    diff_models2 = set(scene_df["model"]) - set(character_df["model"])
    if len(diff_models2) > 0:
        print(f"Models in scene but not in character: {diff_models2}")

    
    # Rename the score columns to be more descriptive
    combined_df = combined_df.rename(columns={
        "final_avg_score_character": "final_avg_score_character",
        "final_avg_score_scene": "final_avg_score_scene"
    })
    
    # Calculate combined average score
    combined_df["final_avg_score_combined"] = (combined_df["final_avg_score_character"] + combined_df["final_avg_score_scene"]) / 2
    
    # Add beautiful names
    combined_df["beautiful_name"] = combined_df["model"].map(model_mapping)
    
    # Calculate combined metrics for each metric present in both subsets
    metric_cols = {}
    for col in combined_df.columns:
        if col.startswith("metric_") and "_character" in col:
            base_metric = col.replace("_character", "")
            scene_col = base_metric + "_scene"
            if scene_col in combined_df.columns:
                combined_metric = base_metric + "_combined"
                combined_df[combined_metric] = (combined_df[col] + combined_df[scene_col]) / 2
                metric_cols[combined_metric] = base_metric.replace("metric_", "") #+ " (combined)"
    
    # Sort by combined score
    combined_df = combined_df.sort_values("final_avg_score_combined", ascending=False)
    
    # Add rank column
    combined_df.insert(0, "rank", range(1, len(combined_df) + 1))
    
    return combined_df, metric_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpbench_only", action="store_true",
                        help="Limit to RPBench models only")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--model_config", type=str, default="./config/api_config.yaml")
    args = parser.parse_args()

    rp_bench_models = [
        "claude-3.5-sonnet-20240620",
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
        "gemini-1.5-pro-001",
        "higgs-llama-3-70b",
    ]

    # Load model mappings
    model_mapping = load_model_mappings(args.model_config)

    # Load results for both subsets
    character_results = load_results(args.result_dir, "character",
                                   model_filter=rp_bench_models if args.rpbench_only else None)
    scene_results = load_results(args.result_dir, "scene",
                               model_filter=rp_bench_models if args.rpbench_only else None)
    
    # Create individual leaderboards
    char_df, char_metrics = create_leaderboard(character_results, model_mapping, "character")
    scene_df, scene_metrics = create_leaderboard(scene_results, model_mapping, "scene")
    
    # Create combined leaderboard
    combined_df, combined_metrics = create_combined_leaderboard(char_df, scene_df, model_mapping)
    
    # Format display dataframes
    char_display = format_display_df(char_df, char_metrics)
    scene_display = format_display_df(scene_df, scene_metrics)
    combined_display = format_display_df(combined_df, combined_metrics)
    combined_display2 = format_display_df2(combined_df, combined_metrics)
    # Create percentage dataframes
    char_percent = create_percentage_df(char_df, char_metrics)
    scene_percent = create_percentage_df(scene_df, scene_metrics)
    combined_percent = create_percentage_df(combined_df, combined_metrics)
    
    # Save outputs for each subset
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Character results
    char_df.to_csv(os.path.join(args.result_dir, "leaderboard_character_full.csv"), index=False)
    char_display.to_csv(os.path.join(args.result_dir, "leaderboard_character_display.csv"), index=False)
    char_percent.to_csv(os.path.join(args.result_dir, "leaderboard_character_percent.csv"), index=False)
    
    # Scene results
    scene_df.to_csv(os.path.join(args.result_dir, "leaderboard_scene_full.csv"), index=False)
    scene_display.to_csv(os.path.join(args.result_dir, "leaderboard_scene_display.csv"), index=False)
    scene_percent.to_csv(os.path.join(args.result_dir, "leaderboard_scene_percent.csv"), index=False)
    
    # Combined results
    combined_df.to_csv(os.path.join(args.result_dir, "leaderboard_combined_full.csv"), index=False)
    combined_display.to_csv(os.path.join(args.result_dir, "leaderboard_combined_display.csv"), index=False)
    combined_display2.to_csv(os.path.join(args.result_dir, "leaderboard_combined_display2.csv"), index=False)
    combined_percent.to_csv(os.path.join(args.result_dir, "leaderboard_combined_percent.csv"), index=False)
    # save combined ranking of just rankd and model
    combined_df[["rank", "model"]].to_csv(os.path.join(args.result_dir, "leaderboard_combined_rank.csv"), index=False)
    CRITERIA = [
        "Contextual_Alignment",
        "Character_Consistency",
        "Descriptive_Depth",
        "Role_Specific_Knowledge",
        "Engagement_and_Collaboration",
        "Creativity_and_Emotional_Nuance",
    ]
    # save combined sorted for each attribute
    # remove rank
    combined_df = combined_df.drop(columns=["rank"])
    for cri in CRITERIA:
        combined_df.sort_values(f"metric_{cri}_combined", ascending=False).to_csv(os.path.join(args.result_dir, f"leaderboard_combined_{cri}.csv"), index=True)
    
    # Print results
    print("\nCharacter Leaderboard:")
    print(char_display.head())
    print("\nScene Leaderboard:")
    print(scene_display.head())
    print("\nCombined Leaderboard:")
    print(combined_display.head())

if __name__ == "__main__":
    main()