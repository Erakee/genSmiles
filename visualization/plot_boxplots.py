import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(csv_file):
    """Load data from CSV and return DataFrame"""
    df = pd.read_csv(csv_file)
    return df

def set_science_style():
    """Set style for scientific publication"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'Arial',
        'font.size': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 20
    })

def plot_normalized_boxplots(df, output_dir='./plots'):
    """Create normalized boxplots for all properties in one plot"""
    # 获取需要处理的属性
    properties = [col for col in df.columns if col != 'smiles']
    
    # 数据归一化
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[properties])
    normalized_df = pd.DataFrame(normalized_data, columns=properties)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6), dpi=500)
    
    # 绘制箱型图
    colors = ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3', '#FF7F00']
    positions = np.arange(len(properties))
    bplot = ax.boxplot([normalized_df[prop] for prop in properties],
                       positions=positions,
                       patch_artist=True,
                       medianprops=dict(color="black", linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='gray', 
                                     markersize=4, alpha=0.5))
    
    # 设置箱型图颜色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置坐标轴
    ax.set_xticklabels(properties, rotation=30, ha='right')
    ax.set_ylabel('Normalized Value')
    
    # 设置坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加网格线
    # ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'{output_dir}/normalized_boxplots.png', 
                bbox_inches='tight', dpi=500, facecolor='white')
    plt.close()

def main():
    # 加载数据
    csv_file = 'D:/Project/genSmiles/dataset/train_copy.csv'
    df = load_and_process_data(csv_file)
    
    # 设置绘图样式
    set_science_style()
    
    # 创建归一化的箱型图
    plot_normalized_boxplots(df)

if __name__ == '__main__':
    main() 