import utils, os, pickle
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from tqdm import tqdm

# 读取训练集SMILES
chembl_smiles = []
chembl_fps = []
with open(utils.config['fname_dataset'], 'r') as f:
    chembl_smiles = set([sm.strip() for sm in f.readlines()])
### todo
# 研究http://www.dalkescientific.com/writings/diary/archive/2020/10/02/using_rdkit_bulktanimotosimilarity.html 
# 生成 'dataset/fps.pkl'
# 搞懂评价标准，如何评价生成分子的质量 
# 使用含能分子，训练vae， 采样， 分析
# 读取预计算的分子指纹
with open(utils.config['fname_fps'], 'rb') as f:
    chembl_fps = pickle.load(f)

def analysis(fname, train_data=False):
    """分析生成的分子文件
    
    Args:
        fname: SMILES文件路径
        train_data: 是否是训练数据
    """
    print('analysis "%s"' % (fname,))
    simil, weight = [], []  # 存储相似度和分子量
    if not train_data:
        # 读取生成的SMILES
        with open(fname, 'r') as f:
            reinforce_smiles = [sm.strip() for sm in f.readlines()]
            # 过滤无效的SMILES
            reinforce_smiles = [sm for sm in reinforce_smiles if utils.isValidSmiles(sm)]
            # 获取唯一的SMILES
            unique_reinforce_smiles = set(reinforce_smiles)
            # 获取与训练集不重叠的SMILES
            new_reinforce_smiles = unique_reinforce_smiles.difference(chembl_smiles)
    else:
        new_reinforce_smiles = chembl_smiles
    # 遍历新分子，计算相似度和分子量
    for sm in tqdm(new_reinforce_smiles):
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            if not train_data:
                # 计算生成分子与训练集的最大相似度
                fp = utils.SimilEvaluator.genFP(mol)
                simil.append(max(BulkTanimotoSimilarity(fp, chembl_fps)))
            # 计算分子量
            weight.append(Descriptors.MolWt(mol))
    simil, weight = np.array(simil), np.array(weight)
    if not train_data:
        # 打印评估指标:
        # - valid: 生成的SMILES能被RDKit正确解析为分子的比例
        # - unique: 生成的有效分子中唯一SMILES的比例
        # - new: 生成的分子与训练集没有重叠的比例/新SMILES的比例
        # - simil_mean/std: 与训练集的平均/标准差相似度，适中最好，太高说明过拟合，太低说明偏离化学空间
        # - weight_mean/std: 分子量的平均值/标准差，应该与训练集接近，用于检查是否保持了基本的分子特征 
        # 一次测试运行的结果输出valid= 61.9 unique= 61.9 new= 61.7 simil_mean= 0.546 simil_std= 0.102 weight_mean=  413.2 weight_std=  107.8
        # 上次跑的是原数据集，重新用em数据集训练并sample后跑的valid= 64.2 unique= 62.6 new= 56.0 simil_mean= 0.515 simil_std= 0.171 weight_mean=  197.5 weight_std=   36.8
        print("%s: valid= %.1f unique= %.1f new= %.1f simil_mean= %.3f simil_std= %.3f weight_mean= %6.1f weight_std= %6.1f" % 
              (fname, len(reinforce_smiles) / 100, len(unique_reinforce_smiles) / 100, 
               len(new_reinforce_smiles) / 100, simil.mean(), simil.std(), weight.mean(), weight.std()))
    else:
        # 对于训练数据只打印分子量统计
        print("%s: weight_mean= %6.1f weight_std= %6.1f" % (fname, weight.mean(), weight.std()))

# 分析训练数据
analysis(utils.config['fname_dataset'], train_data=True)
# 分析不同模型生成的分子
for fname in ('rnn.smi', 'vae.smi'):  #, 'reinforce.smi', 'reinvent.smi'):
    # fname ='vae.smi'
    fname = os.path.join(utils.config['sampled_dir'], fname)
    analysis(fname)
