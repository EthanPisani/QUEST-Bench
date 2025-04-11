import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from textwrap import wrap
import os

def create_leaderboard_graphic(df, title="Model Leaderboard", output_file="leaderboard.png"):
    """Create a styled leaderboard graphic from dataframe"""
    # Make copy and clean data
    df = df.copy().reset_index(drop=True)
    
    # Remove 'Rank' column if it exists (we'll use our own rank column)
    if 'Rank' in df.columns:
        df.drop(columns=['Rank'], inplace=True)
    if 'rank' in df.columns:
        df.drop(columns=['rank'], inplace=True)
        
    # Add our own rank column
    df.insert(0, 'rank', df.index + 1)
    
    # Clean column names - remove underscores, capitalize, and wrap
    def clean_and_wrap(col_name, max_length=13):
        cleaned = col_name.replace('_', ' ').title()
        return '\n'.join(wrap(cleaned, max_length))
    
    df.columns = [clean_and_wrap(col) for col in df.columns]
    
    # Wrap long model names (max 15 chars per line, max 2 lines)
    if 'Model' in df.columns:
        df['Model'] = df['Model'].apply(lambda x: '\n'.join(wrap(str(x), 27))[:60])
    
    # Create colormaps for each attribute
    attribute_colors = {
        'Average Score': plt.cm.Greens,
        'Character Score': plt.cm.Reds,
        'Scene Score': plt.cm.YlOrBr,
        'Contextual Alignment': plt.cm.Blues,
        'Character Consistency': plt.cm.Purples,
        'Descriptive Depth': plt.cm.Blues,
        'Role Specific Knowledge': plt.cm.Purples,
        'Engagement And Collaboration': plt.cm.Blues,
        'Creativity And Emotional Nuance': plt.cm.Purples,
    }
    
    # Medal colors for top 3 ranks
    medal_colors = {
        1: {'bg': '#FFF2CC', 'text': '#D4A017'},  # Gold
        2: {'bg': '#F2F2F2', 'text': '#7F7F7F'},  # Silver 
        3: {'bg': '#F8E5D6', 'text': '#B87333'},  # Bronze
    }
    
    # Create figure with dynamic height
    fig_height = max(6, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    
    # Create title
    plt.title(title, fontsize=18, pad=20, fontweight='bold', color='#2E2E2E')
    
    # Create table with adjusted column widths
    col_widths = [0.08] + [0.3 if 'Model' in col else 0.12 for col in df.columns[1:]]
    table = plt.table(cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colWidths=col_widths,
                    colColours=['#f8f9fa']*len(df.columns),
                    bbox=[0, 0, 1, 0.9])
    
    # Style cells
    for i, (idx, row) in enumerate(df.iterrows()):
        for j, col in enumerate(df.columns):
            cell = table[i+1, j]
            
            # Header style
            if i == -1:  # Header row
                cell.set_text_props(weight='bold', color='white', fontsize=10)
                cell.set_facecolor('#4F81BD')
                cell.set_height(0.15)
                continue
            
            # Medal coloring for top 3 ranks
            if idx < 3 and 'rank' in col.lower():
                cell.set_facecolor(medal_colors[idx+1]['bg'])
                cell.get_text().set_color(medal_colors[idx+1]['text'])
                cell.set_text_props(weight='bold', fontsize=10)
            
            # Model column styling
            elif 'Model' in col:
                cell.set_facecolor('#FFFFFF')
                cell.set_text_props(ha='left', weight='bold', fontsize=10)
                cell.set_width(0.3)
            
            # Attribute columns with color grading
            else:
                for attribute, cmap in attribute_colors.items():
                    # print(f"Checking attribute: {attribute} in column: {col}")
                    
                    # check if first 4 chars match and last 4 chars match
                    if attribute[:4].lower() in col.lower() and attribute[-4:].lower() in col.lower():
                        # print(f"Applying color for attribute: {attribute} in column: {col}")
                        value = float(row[col])
                        min_val = df[col].min()
                        max_val = df[col].max()
                        norm_value = (value - min_val) / (max_val - min_val + 1e-10)
                        
                        # Use lower range of colormap for better visibility and scale so small differences are visible
                        color = cmap(0.1 + 0.8 * norm_value)
                        cell.set_facecolor(color)
                        
                        # Set text color based on background brightness
                        if norm_value > 0.6:
                            cell.set_text_props(color='white', fontsize=9)
                        else:
                            cell.set_text_props(color='#2E2E2E', fontsize=9)
                        break
                else:
                    # Default style for other columns
                    cell.set_facecolor('#FFFFFF')
                    cell.set_text_props(fontsize=9)
    
    # Adjust cell padding and heights
    table.auto_set_font_size(False)
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_height(0.15)
        else:
            cell.set_height(0.08)
        cell.PAD = 0.05
    
    # Save and show
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Leaderboard graphic saved to {output_file}")

def create_combined_visualization(out_dir="results"):
    """Create visualization with combined scores and attributes"""
    # Load the data
    display_df = pd.read_csv(f"{out_dir}/leaderboard_combined_display.csv")
    percent_df = pd.read_csv(f"{out_dir}/leaderboard_combined_percent.csv")

    
    
    # Save the display dataframe
    os.makedirs("results", exist_ok=True)
    display_df.to_csv(f"{out_dir}/leaderboard_visualization.csv", index=False)
    
    # Create the visualization
    create_leaderboard_graphic(
        display_df,
        title=f"QUESTBench-ITER Combined Benchmark Leaderboard\n{pd.Timestamp.now().strftime('%Y-%m-%d')}",
        output_file=f"{out_dir}/combined_leaderboard.png"
    )
    
    
    
    create_leaderboard_graphic(
        percent_df,
        title=f"QUESTBench-ITER Combined Benchmark Leaderboard (%)\n{pd.Timestamp.now().strftime('%Y-%m-%d')}",
        output_file=f"{out_dir}/combined_leaderboard_percent.png"
    )
import argparse
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate leaderboard graphics")
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="results", 
        help="Output directory for the leaderboard graphics"
    )
    args = parser.parse_args()
    out_dir = args.out_dir

    # Create the combined visualization
    create_combined_visualization(out_dir=out_dir)
    
    # Also create individual visualizations
    data = pd.read_csv(f"{out_dir}/leaderboard_character_display.csv")
    df = pd.DataFrame(data)
    
    create_leaderboard_graphic(
        df, 
        title=f"QUESTBench-ITER Character Benchmark Leaderboard \n{pd.Timestamp.now().strftime('%Y-%m-%d')}",
        output_file=f"{out_dir}/character_leaderboard.png"
    )
    
    data_percent = pd.read_csv(f"{out_dir}/leaderboard_character_percent.csv")
    df_percent = pd.DataFrame(data_percent)

    create_leaderboard_graphic(
        df_percent, 
        title=f"QUESTBench-ITER Character Benchmark Leaderboard (%)\n{pd.Timestamp.now().strftime('%Y-%m-%d')}",
        output_file=f"{out_dir}/character_leaderboard_percent.png"
    )
    
    # Scene leaderboards
    scene_data = pd.read_csv(f"{out_dir}/leaderboard_scene_display.csv")
    scene_df = pd.DataFrame(scene_data)
    
    create_leaderboard_graphic(
        scene_df, 
        title=f"QUESTBench-ITER Scene Benchmark Leaderboard \n{pd.Timestamp.now().strftime('%Y-%m-%d')}",
        output_file=f"{out_dir}/scene_leaderboard.png"
    )
    
    scene_percent = pd.read_csv(f"{out_dir}/leaderboard_scene_percent.csv")
    scene_df_percent = pd.DataFrame(scene_percent)

    create_leaderboard_graphic(
        scene_df_percent, 
        title=f"QUESTBench-ITER Scene Benchmark Leaderboard (%)\n{pd.Timestamp.now().strftime('%Y-%m-%d')}",
        output_file=f"{out_dir}/scene_leaderboard_percent.png"
    )