from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np

def visualize_molecules(smiles_list, n_cols=5, img_size=(300, 300), save_path=None):
    """
    Visualize a list of molecules in a grid layout
    
    Args:
        smiles_list (list): List of SMILES strings
        n_cols (int): Number of columns in the grid
        img_size (tuple): Size of each molecule image (width, height)
        save_path (str): Path to save the combined image (optional)
    """
    # Convert SMILES to RDKit molecules
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Generate 2D coordinates for the molecule
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
        else:
            print(f"Warning: Could not parse SMILES: {smi}")
    
    if not mols:
        print("No valid molecules to display")
        return
    
    # Calculate grid dimensions
    n_mols = len(mols)
    n_rows = int(np.ceil(n_mols / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    
    # Draw each molecule
    for idx, mol in enumerate(mols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        img = Draw.MolToImage(mol, size=img_size)
        ax.imshow(img)
        ax.axis('off')
        # Add SMILES as title (optional)
        # ax.set_title(Chem.MolToSmiles(mol), fontsize=8, wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# 测试用的SMILES列表
smiles_list = [
    'CC1=NC(O)=NC(=N)C2=C10N=NN2C(N)=N',
    'CC(O)(C(N)=CN1N=NOC(=N1)[N+]([O-])=O',
    'OC1=C2CC(=N1)N=C2C(=NN12)C=O',
    'NN=C1N=C(CN=C)OC=C1[N+]([O-])=O', 
    'CC1=CN=C2OC(=N)N=C(O)OC(O)=C1[N+]([O-])=O',
    'CN1N=C(ON=C1[N+]([O-])=O)C1(CO)C(=O)O',
    'CONCC1=NC=C(N2C(=O)N2)=N1',
    'CC1=NNC(=O)N=C1[N+]([O-])=O',
    'NC1C2NC(=O)N=C1N1CC22',
    'NN1N=NC(=C1O)[N+]([O-])=O',
    'CC1=NC(=O)ON=CC(=N1)[N+]([O-])=O',
    'CC(=O)OC1=C(CN(NOC3)C=O)ON=N1',
    'NN(C=C)C1=C(CON=N1)[N+]([O-])=O',
    'CC1=C(O)N2N=NC(=O)C(O)=C2N=C1N=CO2',
    'CC1(CCCN=CC(OC=O)=N1)[N+]([O-])=O',
    'CN1N=NN=C1[N+]([O-])=O',
    'ON1C=C2NCC(=N1)[N+]([O-])=O',
]

def main():
    # 可以从文件读取SMILES
    # with open('path/to/your.smi', 'r') as f:
    #     smiles_list = [line.strip().split()[0] for line in f]
    
    visualize_molecules(
        smiles_list,
        n_cols=5,
        img_size=(300, 300),
        save_path='molecules_grid.png'
    )

if __name__ == "__main__":
    main()