import json
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm

def extract_data(input_data, data_type="scene"):
    """
    Extracts all information from the evaluation runs and saves it to a JSON file.
    
    Args:
        input_data (dict): The loaded JSON data containing evaluation runs
        output_filename (str): Filename to save the extracted data
    """
    extracted_data = {
        "model": input_data["model"],
        "num_runs": input_data["num_runs"],
        "final_avg_score":  0,
        "final_metrics": {},
        "runs": []
    }
    
    for run in input_data["runs"]:
        run_data = {
            "run": run["run"],
            "average_score":0,
            "metrics": {},
            "results": []
        }
        
        for result in run["results"]:
            result_data = {

                "request_type": result["request_type"],
                "user_input": result["user_input"],
                "response": result["response"],
                "scores": {},
                "total_score": 0
            }
            if data_type == "scene":
                result_data["scene_context"] = result["scene_context"]
            elif data_type == "character":
                result_data["npc_profile"] = result["npc_profile"]  # Add character profile
            run_data["results"].append(result_data)
        
        extracted_data["runs"].append(run_data)
    
    return extracted_data

def save_extracted_data(extracted_data, output_filename):
    """
    Saves the extracted data to a JSON file.
    
    Args:
        extracted_data (dict): The extracted data to save
        output_filename (str): Filename to save the extracted data
    """
    with open(output_filename, 'w') as f:
        json.dump(extracted_data, f, indent=4)

def extract(input_dir="results/character", output_dir="responses/character", data_type="scene"):
    """
    Process all benchmark JSON files in the input directory and save extracted data to output directory.
    
    Args:
        input_dir (str): Directory containing input JSON files
        output_dir (str): Directory to save the extracted JSON files
    """
    for file in tqdm(sorted(glob(os.path.join(input_dir, "benchmark_*.json")))):
        with open(file, 'r') as f:
            input_json = json.load(f)
        
        # Extract and save the data
        extracted = extract_data(input_json, data_type)
        output_filename = os.path.join(output_dir, os.path.basename(file))
        save_extracted_data(extracted, output_filename)

import argparse
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Extract and save evaluation data from JSON files.")
    parser.add_argument("--input_dir", type=str, default="results/character", help="Directory containing input JSON files.")
    parser.add_argument("--output_dir", type=str, default="responses/character", help="Directory to save the extracted JSON files.")
    parser.add_argument("--type", type=str, default="scene", choices=["scene", "character"], help="Type of data to extract (scene or character).")
    
    args = parser.parse_args()
    # make dir if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    extract(args.input_dir, args.output_dir, args.type)