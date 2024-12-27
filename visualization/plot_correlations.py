import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        'ytick.labelsize': 16
    })

def plot_correlation_matrix(df, output_dir='./plots'):
    """Create publication-quality correlation matrix heatmap"""
    # 获取需要处理的属性
    properties = [col for col in df.columns if col != 'smiles']
    
    # 计算相关性矩阵
    corr_matrix = df[properties].corr()
    
    # 创建图形
    plt.figure(figsize=(10, 8), dpi=500)
    
    # 创建掩码（只显示下三角）
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # 绘制热图
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'shrink': .8},
                annot_kws={'size': 16},
                linewidths=0.5,
                linecolor='white')
    
    # 设置标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 添加属性单位到标签
    labels = []
    for prop in properties:
        if prop == 'electronic_energy':
            labels.append('Electronic Energy')
        elif prop == 'HOMO_LUMO_gap':
            labels.append('HOMO-LUMO Gap')
        elif prop == 'dipole_moment':
            labels.append('Dipole Moment')
        elif prop == 'crystal_density':
            labels.append('Crystal Density')
        else:  # heat_of_formation
            labels.append('Heat of Formation')
    
    plt.xticks(np.arange(len(properties)) + 0.5, labels)
    plt.yticks(np.arange(len(properties)) + 0.5, labels)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png',
                bbox_inches='tight', dpi=500, facecolor='white')
    plt.close()

def plot_pairplot(df, output_dir='./plots'):
    """Create publication-quality pairplot"""
    # 获取需要处理的属性
    properties = [col for col in df.columns if col != 'smiles']
    
    # 创建pairplot
    g = sns.pairplot(df[properties],
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6},
                     diag_kws={'color': '#2166AC', 'linewidth': 2},
                     corner=True)  # 只显示下三角
    
    # 设置图形大小和DPI
    g.fig.set_size_inches(15, 15)
    g.fig.set_dpi(500)
    
    # 修改标签（添加单位）
    for i, prop in enumerate(properties):
        for j in range(i+1):
            if g.axes[i,j] is not None:
                # 设置y轴标签（只在最左边显示）
                if j == 0:
                    if prop == 'electronic_energy':
                        g.axes[i,j].set_ylabel('Electronic Energy')
                    elif prop == 'HOMO_LUMO_gap':
                        g.axes[i,j].set_ylabel('HOMO-LUMO Gap')
                    elif prop == 'dipole_moment':
                        g.axes[i,j].set_ylabel('Dipole Moment')
                    elif prop == 'crystal_density':
                        g.axes[i,j].set_ylabel('Crystal Density')
                    else:  # heat_of_formation
                        g.axes[i,j].set_ylabel('Heat of Formation')
                
                # 设置x轴标签（只在最底部显示）
                if i == len(properties)-1:
                    if properties[j] == 'electronic_energy':
                        g.axes[i,j].set_xlabel('Electronic Energy')
                    elif properties[j] == 'HOMO_LUMO_gap':
                        g.axes[i,j].set_xlabel('HOMO-LUMO Gap')
                    elif properties[j] == 'dipole_moment':
                        g.axes[i,j].set_xlabel('Dipole Moment)')
                    elif properties[j] == 'crystal_density':
                        g.axes[i,j].set_xlabel('Crystal Density')
                    else:  # heat_of_formation
                        g.axes[i,j].set_xlabel('Heat of Formation')
                else:
                    g.axes[i,j].set_xlabel('')
                
                # 移除网格线并简化样式
                g.axes[i,j].grid(False)
                g.axes[i,j].spines['top'].set_visible(False)
                g.axes[i,j].spines['right'].set_visible(False)
    
    # 调整布局以防止截断
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'{output_dir}/pairplot.png',
                bbox_inches='tight', dpi=500, facecolor='white',
                pad_inches=0.5)  # 增加边距
    plt.close()

def main():
    # 加载数据
    csv_file = 'D:/Project/genSmiles/dataset/train_copy.csv'
    df = load_and_process_data(csv_file)
    
    # 设置绘图样式
    set_science_style()
    
    # 创建相关性矩阵和配对图
    plot_correlation_matrix(df)
    plot_pairplot(df)

if __name__ == '__main__':
    main() 