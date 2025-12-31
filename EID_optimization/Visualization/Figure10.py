import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
csv_path = r"C:\Users\wangx\Desktop\network\omission_evaluation_final_summary.csv"
df = pd.read_csv(csv_path)

# Output directory for plots
output_dir = r"C:\Users\wangx\Desktop\network\removal_plots"
os.makedirs(output_dir, exist_ok=True)

# Get unique values
sentinels = sorted(df['num_sentinels'].unique())
types = sorted(df['type'].unique())
strategies = sorted(df['strategy'].unique())

# Define network order for subplots
network_order = ['Modular', 'Scale-free', 'University', 'High school', 'Facebook', 'Wildbird']
subplot_labels_roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi']

# Color scheme for strategies
strategy_colors = {
    "RFSM" : "#BC3C29FF",
    "Greedy" : "#0072B5FF",
    "GA" : "#E18727FF",
    "Global" : "#20854EFF",
    "Random" : "#FFDC91FF",
    "Modular" : "#7876B1FF"
}

# Line styles for strategies
strategy_styles = {
    'RFSM': '-',
    'GA': '-',
    'Global': '-',
    'Greedy': '-',
    'Modular': '-',
    'Random': '--'
}

# Marker styles for strategies
strategy_markers = {
    'RFSM': 'o',
    'GA': 'o',
    'Global': 'o',
    'Greedy': 'o',
    'Modular': 'o',
    'Random': 'o'
}

print(f"Sentinel counts: {sentinels}")
print(f"Types: {types}")
print(f"Strategies: {strategies}")

# Generate 3 figures (one for each sentinel count, combining both types)
for num_sentinel in sentinels:
    # Create figure with 2x6 subplots (2 types × 6 networks = 2 rows × 6 columns)
    fig, axes = plt.subplots(2, 6, figsize=(36, 12), sharex=True)
    
    # To collect handles and labels for shared legend
    handles, labels = None, None
    
    # Plot for each type (edge and node)
    for type_idx, omission_type in enumerate(types):
        # Add section label (A or B) on the left side
        section_label = chr(65 + type_idx)  # 'A' or 'B'
        
        # Add section title to the left of each row (增大4: 24 -> 28)
        fig.text(0.05, 1.0 - (type_idx * 0.48), 
                f'{section_label}. {omission_type.capitalize()} removal',
                fontsize=28, fontweight='bold', 
                verticalalignment='top')
        
        # Plot each network in specified order (all 6 networks in one row)
        for idx, network in enumerate(network_order):
            ax = axes[type_idx, idx]
            
            # Filter data for this network, sentinel count, and type
            data_subset = df[(df['network_name'] == network) & 
                           (df['num_sentinels'] == num_sentinel) & 
                           (df['type'] == omission_type)]
            
            # Plot each strategy with shaded error bars
            for strategy in strategies:
                strategy_data = data_subset[data_subset['strategy'] == strategy]
                
                if len(strategy_data) > 0:
                    # Sort by omission proportion
                    strategy_data = strategy_data.sort_values('omission_pct')
                    
                    x = strategy_data['omission_pct']
                    y = strategy_data['surveillance_performance_mean']
                    std = strategy_data['surveillance_performance_std']
                    
                    # Plot line
                    line = ax.plot(x, y,
                           label=strategy,
                           color=strategy_colors[strategy],
                           linestyle=strategy_styles[strategy],
                           marker=strategy_markers[strategy],
                           markersize=6,
                           linewidth=2.5,
                           alpha=0.8)
                    
                    # Add shaded error region (mean ± std)
                    ax.fill_between(x, y - std, y + std,
                                   color=strategy_colors[strategy],
                                   alpha=0.15)
            
            # Get handles and labels from first subplot for shared legend
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            
            # Add subplot label (i-vi) with network name at the top (增大4: 24 -> 28)
            ax.set_title(f'{subplot_labels_roman[idx]}. {network}',
                        fontsize=28, fontweight='bold', pad=3, loc='left')
            
            # Remove grid
            ax.grid(False)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Increase tick label size (增大4: 14 -> 18)
            ax.tick_params(axis='both', which='major', labelsize=18)
            
            # Set x-axis limits
            ax.set_xlim(0, 100)
            
            # Only add y-label to leftmost subplots (增大4: 20 -> 24)
            if idx == 0:
                ax.set_ylabel('Surveillance performance\n(% cases prevented)', fontsize=24)
    
    # Add shared x-axis label for bottom row (增大4: 20 -> 24)
    for col in range(6):
        axes[1, col].set_xlabel('Removal proportion (%)', fontsize=24)
    
    # Add shared legend at the bottom of the figure (增大4: 18 -> 22)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
              ncol=len(strategies), fontsize=22, framealpha=0.9)
    
    # Adjust layout
    plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.08, 
                       hspace=0.25, wspace=0.1)
    
    # Save figure
    filename = f"performance_{num_sentinel}sentinels_combined_2x6.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Also save as PDF for better quality
    filepath_pdf = os.path.join(output_dir, filename.replace('.png', '.pdf'))
    plt.savefig(filepath_pdf, bbox_inches='tight')
    
    plt.close()
