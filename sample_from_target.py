from tokens import Tokenizer
import os
import torch
import utils
import vae
import numpy as np

def sample_from_smiles(smiles, n_samples=10, sigma=0.1):
    """从给定SMILES字符串生成相似的新分子
    
    Args:
        smiles: 输入SMILES字符串
        n_samples: 生成样本数量
        sigma: 隐空间扰动强度
    Returns:
        generated: 生成的有效SMILES列表
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 获取tokenizer
    tokenizer = utils.get_tokenizer()
    
    # 加载VAE模型
    model = vae.VAE(**utils.config['vae_param'],
                    encoder_state_fname=utils.config['fname_vae_encoder_parameters'],
                    decoder_state_fname=utils.config['fname_vae_decoder_parameters'],
                    device=device)
    model.encoder.loadState()
    model.decoder.loadState()
    model.encoder.eval()
    model.decoder.eval()

    # 将SMILES转换为one-hot编码
    tokens = tokenizer.tokenize([smiles], useTokenDict=True)
    num_vectors = tokenizer.getNumVector(tokens)
    X = torch.zeros((1, utils.config['maxLength'], 
                    tokenizer.getTokensSize() - 2), 
                    dtype=torch.float32,
                    device=device)
    for j, n in enumerate(num_vectors[0]):
        X[0, j, n - 2] = 1
    if j + 1 < utils.config['maxLength']:
        X[0, j + 1:, 0] = 1

    # 编码到隐空间
    with torch.no_grad():
        z, mu, _ = model.encoder(X)
        
        # 多次采样生成新分子
        generated = []
        while len(generated) < n_samples:
            # 对隐空间表示添加随机扰动
            z_new = mu + torch.randn_like(mu) * sigma
            
            # 解码生成新分子
            y = model.decoder(z_new, None, freerun=True)
            num_vectors = y.argmax(dim=2) + 2
            
            # 转换为SMILES
            new_smiles = tokenizer.getSmiles(num_vectors.cpu())[0]
            if utils.isValidSmiles(new_smiles) and new_smiles not in generated:
                generated.append(new_smiles)
                
    return generated

def save_generated_smiles(smiles_list, output_file):
    """保存生成的SMILES到文件
    
    Args:
        smiles_list: SMILES字符串列表
        output_file: 输出文件路径
    """
    utils.mkdir_multi(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(smiles_list))
        f.write('\n')

if __name__ == "__main__":
    # 示例用法
    target_smile_name = "HMX"
    # input_smiles = "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"  # RDX
    input_smiles = "C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"  # HMX
    n_samples = 10
    sigma = 0.1
    
    print(f"Input SMILES: {input_smiles}")
    print(f"Generating {n_samples} similar molecules (sigma={sigma})...")
    
    generated_smiles = sample_from_smiles(input_smiles, n_samples=n_samples, sigma=sigma)
    
    print("\nGenerated similar molecules:")
    for i, smi in enumerate(generated_smiles, 1):
        print(f"{i}. {smi}")
        
    # 保存生成的分子
    output_file = os.path.join(utils.config['target_sample_dir'], f'generated_from_{target_smile_name}.smi')
    save_generated_smiles(generated_smiles, output_file)
    print(f"\nSaved generated SMILES to {output_file}") 