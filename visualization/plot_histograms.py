import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

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
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    })

def plot_histogram(df, property_name, output_dir='./plots'):
    """Create publication-quality histogram for a single property"""
    # 设置图形大小（3:4比例）
    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    
    # 获取数据
    data = df[property_name]
    
    # 设置合适的bin数量
    if property_name == 'electronic_energy':
        bins = np.linspace(-1100, -600, 20)  # 电子能量范围约为-1000到-500
    elif property_name == 'HOMO_LUMO_gap':
        bins = np.linspace(1, 8, 20)  # HOMO-LUMO gap范围约为2到7
    elif property_name == 'dipole_moment':
        bins = np.linspace(-2, 14, 20)  # 偶极矩范围约为0到12
    elif property_name == 'crystal_density':
        bins = np.linspace(1.3, 2.0, 20)  # 晶体密度范围约为1.4到1.9
    else:  # heat_of_formation
        bins = np.linspace(-450, 250, 20)  # 生成热范围约为-400到200
    
    # Scientific color palette
    colors = ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3', '#FF7F00']
    properties = [col for col in df.columns if col != 'smiles']
    color_idx = properties.index(property_name)
    
    # 绘制直方图
    counts, edges, patches = ax.hist(data, bins=bins,
                                   color=colors[color_idx % len(colors)],
                                   alpha=0.7, density=True,
                                   edgecolor='white', linewidth=0.5)
    
    # 添加核密度估计曲线
    kde = pd.Series(data).plot.kde()
    x = kde.get_lines()[0].get_xdata()
    y = kde.get_lines()[0].get_ydata()
    # 限制KDE曲线的范围与直方图一致
    mask = (x >= bins[0]) & (x <= bins[-1])
    ax.plot(x[mask], y[mask], color='black', linewidth=1.5)
    
    # 设置坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置刻度
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 设置标签
    ax.set_xlabel(property_name)
    if property_name == 'electronic_energy':
        ax.set_ylabel('Hartree')
    elif property_name == 'HOMO_LUMO_gap':
        ax.set_ylabel('eV')
    elif property_name == 'dipole_moment':
        ax.set_ylabel('Debye')
    elif property_name == 'crystal_density':
        ax.set_ylabel('g/cc')
    else:  # heat_of_formation
        ax.set_ylabel('kcal/mol')
    
    # 设置坐标轴范围
    ax.set_xlim(bins[0], bins[-1])
    
    # 保存图片
    plt.tight_layout()
    safe_filename = property_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    plt.savefig(f'{output_dir}/{safe_filename}_hist.png', 
                bbox_inches='tight', dpi=500, facecolor='white')
    plt.close()

def main():
    # 加载数据
    csv_file = 'D:/Project/genSmiles/dataset/train_copy.csv'
    df = load_and_process_data(csv_file)
    
    # 设置绘图样式
    set_science_style()
    
    # 为每个属性创建直方图
    properties = [col for col in df.columns if col != 'smiles']
    for prop in properties:
        plot_histogram(df, prop)

if __name__ == '__main__':
    main() 