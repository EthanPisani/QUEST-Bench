# Quest-Bench

An automated pipeline for evaluating LLMs for role-playing.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the evaluation scripts for each roleplay type. Replace `[model]` with the name of the model you want to evaluate.
You will need to edit api_config.yml for your configuration.

**Scene Evaluation:**

```bash
python run_iter_scene_eval.py --model [model] 
```

**Character Evaluation:**

```bash
python run_iter_character_eval.py --model [model]
```

**Extract Responses:**

After running the evaluation scripts, extract the responses:

```bash
python extract_rankings.py --input_dir results/scene --output_dir responses/scene
python extract_rankings.py --input_dir results/character --output_dir responses/character
```

**Score Responses:**

Finally, score the responses:

```bash
python score_responses.py --input_dir responses/scene --output_dir scored/scene --roleplay_type scene
python score_responses.py --input_dir responses/character --output_dir scored/character --roleplay_type character 
```

## Generating the Leaderboard and Graphic

To generate the leaderboard and graphic:

```bash
python iter_generate_leaderboard_combined.py --result_dir scored
python iter_generate_lead_graphic_combined.py --result_dir scored
```

## View Results

To view the leaderboard:

*   Leaderboard files in the `scored` directory, `leaderboard_character_percent.csv` and `leaderboard_combined_display.csv`.
*   The response files in `/results/character` and `/results/scene`. 

## Acknowledgements

This benchmark is heavily inspired by [RPBench-Auto](https://github.com/boson-ai/RPBench-Auto), [ArenaHard](https://github.com/lm-sys/arena-hard-auto) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). Some code implementations are borrowed from these repositories.
