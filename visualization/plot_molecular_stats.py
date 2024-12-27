import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import sys
import os
from rdkit import Chem

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokens import Tokenizer, getTokenizer

def load_smiles(file_path):
    """Load SMILES from CSV file"""
    df = pd.read_csv(file_path)
    return df['smiles'].tolist()

def analyze_molecules(smiles_list, tokenizer):
    """分析分子中的元素分布和token数"""
    token_counts = Counter()
    tokens_per_mol = []
    
    for smi in smiles_list:
        # 使用项目的tokenizer来分词
        tokens = tokenizer.tokenize([smi])[0]
        tokens_per_mol.append(len(tokens))
        
        # 统计token
        for token in tokens:
            if token not in ['(', ')', '=', '#', '[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-']:
                token_counts[token] += 1
    
    return token_counts, tokens_per_mol

def plot_element_distribution(token_counts, output_dir='./plots'):
    """绘制元素分布图"""
    plt.figure(figsize=(15, 8), dpi=500)
    elements = list(token_counts.keys())
    counts = list(token_counts.values())
    
    # 按计数排序
    sorted_indices = np.argsort(counts)[::-1]
    elements = [elements[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # 绘制条形图
    bars = plt.bar(elements, counts, color='#2166AC', alpha=0.7)
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom',
                fontsize=12)
    
    plt.title('Element Distribution in Dataset', fontsize=20, pad=20)
    plt.xlabel('Elements and Special Tokens', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    
    # 设置样式
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # 添加百分比标签
    total = sum(counts)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height/total*100
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.1f}%',
                ha='center', va='center',
                fontsize=12, color='white')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{output_dir}/element_distribution.png',
                bbox_inches='tight', dpi=500, facecolor='white')
    plt.close()

def plot_token_count_distribution(token_counts, output_dir='./plots'):
    """绘制每个分子的token数分布图"""
    plt.figure(figsize=(10, 6), dpi=500)
    
    # 设置bin的数量
    n_bins = min(int(np.sqrt(len(token_counts))), 30)
    
    # 绘制直方图
    sns.histplot(token_counts, bins=n_bins, color='#2166AC', alpha=0.7,
                 kde=True, line_kws={'color': 'red', 'linewidth': 2})
    
    plt.title('Distribution of Tokens per Molecule', fontsize=20, pad=20)
    plt.xlabel('Number of Tokens', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    
    # 添加统计信息
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    plt.axvline(mean_tokens, color='red', linestyle='--', alpha=0.5,
                label=f'Mean: {mean_tokens:.1f}')
    plt.axvline(median_tokens, color='green', linestyle='--', alpha=0.5,
                label=f'Median: {median_tokens:.1f}')
    plt.legend()
    
    # 设置样式
    plt.grid(linestyle='--', alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{output_dir}/token_count_distribution.png',
                bbox_inches='tight', dpi=500, facecolor='white')
    plt.close()

def count_atoms(smiles):
    """
    Count the actual number of atoms in a SMILES string.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        int: Number of atoms, returns 0 if SMILES is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumAtoms()
    except:
        return 0

def plot_atom_distribution(smi_file, output_dir=None, bins=50):
    """
    Plot the distribution of atom counts in molecules from a SMILES file.
    
    Args:
        smi_file (str): Path to the SMILES file
        output_dir (str): Directory to save the plot (optional)
        bins (int): Number of bins for histogram
    """
    # Read SMILES and count atoms
    atom_counts = []
    valid_count = 0
    total_count = 0
    
    with open(smi_file, 'r') as f:
        for line in f:
            total_count += 1
            smiles = line.strip().split()[0]  # Get SMILES from first column
            atom_count = count_atoms(smiles)
            if atom_count > 0:
                atom_counts.append(atom_count)
                valid_count += 1
    
    # Calculate statistics
    avg_atoms = np.mean(atom_counts)
    median_atoms = np.median(atom_counts)
    std_atoms = np.std(atom_counts)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=bins, edgecolor='black')
    plt.axvline(avg_atoms, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {avg_atoms:.1f}')
    plt.axvline(median_atoms, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_atoms:.1f}')
    
    plt.title(f'Distribution of Atom Counts in Molecules\n'
              f'(Valid molecules: {valid_count}/{total_count}, {valid_count/total_count*100:.1f}%)')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(f'{output_dir}/atom_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\nMolecule Statistics:")
    print(f"Total SMILES: {total_count}")
    print(f"Valid molecules: {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"Average atoms per molecule: {avg_atoms:.2f}")
    print(f"Median atoms per molecule: {median_atoms:.2f}")
    print(f"Standard deviation: {std_atoms:.2f}")
    
    # Return statistics for further use if needed
    return {
        'total_count': total_count,
        'valid_count': valid_count,
        'average_atoms': avg_atoms,
        'median_atoms': median_atoms,
        'std_atoms': std_atoms,
        'atom_counts': atom_counts
    }

def main():
    # 加载数据
    csv_file = 'D:/Project/genSmiles/dataset/train_copy.csv'
    smiles_list = load_smiles(csv_file)
    
    # 初始化tokenizer
    tokenizer = getTokenizer(csv_file)
    
    # 分析分子
    token_counts, tokens_per_mol = analyze_molecules(smiles_list, tokenizer)
    
    # 创建图表
    plot_element_distribution(token_counts)
    plot_token_count_distribution(tokens_per_mol)

if __name__ == '__main__':
    main() 