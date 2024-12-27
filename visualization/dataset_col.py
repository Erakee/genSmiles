import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_and_process_data(csv_file):
    """Load data from CSV and return DataFrame"""
    df = pd.read_csv(csv_file)
    return df

def set_science_style():
    """Set style for scientific publication"""
    plt.style.use('default')  # Reset style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'font.family': 'Arial'
    })

def plot_property_distributions(df, output_dir='D:/Project/genSmiles/plots'):
    """Create distribution plots for all properties"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Scientific color palette
    colors = ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3', '#FF7F00']
    properties = [col for col in df.columns if col != 'smiles']
    
    set_science_style()
    
    for idx, prop in enumerate(properties):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=500)
        fig.suptitle(f'Distribution of {prop}', y=0.95)
        
        # Histogram with KDE
        sns.histplot(data=df, x=prop, bins=30, ax=ax1, color=colors[idx % len(colors)],
                    alpha=0.6, edgecolor='white', linewidth=0.5)
        sns.kdeplot(data=df, x=prop, ax=ax1, color='black', linewidth=1.5)
        
        # Boxplot
        sns.boxplot(data=df, x=prop, ax=ax2, color=colors[idx % len(colors)],
                   saturation=0.7, width=0.4)
        ax2.set_title('Box Plot', pad=10)
        
        # Remove excessive grid lines
        ax1.grid(False)
        ax2.grid(False)
        
        plt.tight_layout()
        safe_filename = prop.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        plt.savefig(os.path.join(output_dir, f'{safe_filename}_distribution.png'), 
                   bbox_inches='tight', facecolor='white')
        plt.close()

def plot_correlation_matrix(df, output_dir='./plots'):
    """Create correlation matrix heatmap"""
    os.makedirs(output_dir, exist_ok=True)
    set_science_style()
    
    numerical_cols = [col for col in df.columns if col != 'smiles']
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(8, 6), dpi=500)
    mask = np.triu(np.ones_like(corr_matrix), k=1)  # 只显示下三角
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r',
                center=0, 
                fmt='.2f',
                square=True,
                cbar_kws={'shrink': .8},
                annot_kws={'size': 8},
                linewidths=0.5,
                linecolor='white')
    
    plt.title('Correlation Matrix of Properties', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), 
                bbox_inches='tight', facecolor='white')
    plt.close()

def plot_pairplot(df, output_dir='./plots'):
    """Create pairplot for all properties"""
    os.makedirs(output_dir, exist_ok=True)
    set_science_style()
    
    numerical_cols = [col for col in df.columns if col != 'smiles']
    
    g = sns.pairplot(df[numerical_cols],
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 'edgecolor': None},
                     diag_kws={'color': '#2166AC', 'linewidth': 1.5},
                     corner=True)  # 只显示下三角
    
    g.fig.set_dpi(500)
    g.fig.suptitle('Pairwise Relationships', y=1.02)
    
    # 移除网格线并简化样式
    for ax in g.axes.flat:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.savefig(os.path.join(output_dir, 'pairplot.png'), 
                bbox_inches='tight', facecolor='white')
    plt.close()

def print_statistics(df):
    """Print basic statistics for each property"""
    numerical_cols = [col for col in df.columns if col != 'smiles']
    
    print("\nBasic Statistics:")
    print("=" * 50)
    for col in numerical_cols:
        print(f"\n{col}:")
        print("-" * 30)
        stats = df[col].describe()
        print(stats)

def main():
    # Load data
    csv_file = 'D:/Project/genSmiles/dataset/train.csv'
    df = load_and_process_data(csv_file)
    
    # Create visualizations
    plot_property_distributions(df)
    plot_correlation_matrix(df)
    plot_pairplot(df)
    print_statistics(df)

if __name__ == '__main__':
    main()