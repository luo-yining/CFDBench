import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tap import Tap
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

# Importações do projeto CFDBench
from models.cfd_vae import CfdVae
from dataset import get_auto_dataset
from train_vae import VaeDataset # Reutilizamos o VaeDataset que já criamos

class InterpretArgs(Tap):
    """Argumentos para interpretar o VAE treinado."""
    data_name: str = "tube_geo"
    data_dir: str = "../data"
    weights_path: str = "../weights/cfd_vae.pt"
    latent_dim: int = 4
    num_samples_for_pca: int = 500
    num_traversal_steps: int = 10 # Número de passos na travessia
    traversal_range: float = 10.0 # Quão longe percorrer (em unidades de desvio padrão)

def main():
    """Função principal para interpretar o VAE."""
    args = InterpretArgs().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Carregar o Modelo VAE Treinado ---
    print("Carregando modelo VAE treinado...")
    model = CfdVae(in_chan=2, out_chan=2, latent_dim=args.latent_dim)
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Modelo carregado com sucesso.")

    # --- 2. Preparar os Dados para Análise ---
    print("Preparando dados para análise do espaço latente...")
    # Carregamos apenas o conjunto de teste para a análise
    _, _, test_data_raw = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
        load_splits=['test']
    )
    assert test_data_raw is not None
    
    test_dataset = VaeDataset(test_data_raw)
    
    # --- 3. Mapear o Espaço Latente e Aplicar PCA ---
    print(f"Codificando {args.num_samples_for_pca} amostras para mapear o espaço latente...")
    latents = []
    with torch.no_grad():
        for i in tqdm(range(min(args.num_samples_for_pca, len(test_dataset)))):
            sample = test_dataset[i].unsqueeze(0).to(device)
            posterior = model.vae.encode(sample).latent_dist
            latents.append(posterior.mean.cpu().reshape(1, -1)) # Achata o latente
            
    all_latents = torch.cat(latents, dim=0).numpy()
    
    print("Calculando Componentes Principais (PCA)...")
    pca = PCA(n_components=args.latent_dim) # Usamos no máximo o número de canais latentes
    pca.fit(all_latents)
    
    # --- 4. Realizar e Visualizar a Travessia do Espaço Latente ---
    print("Realizando a travessia do espaço latente ao longo dos componentes principais...")
    
    # Pegamos uma amostra de referência do meio do conjunto de teste
    reference_idx = len(test_dataset) // 2
    reference_sample = test_dataset[reference_idx].unsqueeze(0).to(device)
    with torch.no_grad():
        reference_latent_mean = model.vae.encode(reference_sample).latent_dist.mean
        
    # Travessia para os 3 componentes principais mais importantes
    num_components_to_show = min(3, pca.n_components_)
    
    fig, axes = plt.subplots(
        num_components_to_show, 
        args.num_traversal_steps, 
        figsize=(args.num_traversal_steps * 2, num_components_to_show * 2.5)
    )
    fig.suptitle("Travessia do Espaço Latente ao Longo dos Componentes Principais", fontsize=16)

    for i in range(num_components_to_show):
        pc = torch.tensor(pca.components_[i], dtype=torch.float32, device=device)
        latent_std = all_latents.std(axis=0)[i] # Desvio padrão ao longo deste PC
        
        for j, alpha in enumerate(np.linspace(-args.traversal_range, args.traversal_range, args.num_traversal_steps)):
            # Vetor latente modificado
            traversed_latent_flat = reference_latent_mean.reshape(1, -1) + alpha * latent_std * pc
            traversed_latent = traversed_latent_flat.reshape(reference_latent_mean.shape)
            
            # Decodificar e plotar
            with torch.no_grad():
                reconstruction = model.vae.decode(traversed_latent).sample
            
            ax = axes[i, j]
            # Plotamos o canal de velocidade U (canal 0)
            ax.imshow(reconstruction[0, 0].cpu().numpy(), cmap='viridis')
            ax.axis('off')
            if j == 0:
                ax.set_title(f"PC {i} (var: {pca.explained_variance_ratio_[i]:.2f})", loc='left', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()

