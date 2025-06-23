import torch
import matplotlib.pyplot as plt

def visualize_prediction(model, dataloader, num_samples, depth_slices ,device):
    model.eval()
      
    with torch.no_grad():
        dataiter=iter(dataloader)
        volumes, masks=next(dataiter)
        volumes, masks=volumes.to(device), masks.to(device)
        
        outputs=model(volumes)
        
        preds=torch.sigmoid(outputs)
        preds=(preds>=0.5).float()
        
        volumes=volumes.cpu().numpy()
        masks=masks.cpu().numpy()
        preds=preds.cpu().numpy()
        
        num_samples=min(num_samples, volumes.size(0))
        
        for i in range(num_samples):
            fig, axes=plt.subplots(len(depth_slices), 3, figsize=(12, 4*len(depth_slices)))
            plt.suptitle(f'Sample {i+1}', fontsize=16, fontweight='bold')
            
            for j, depth in enumerate(depth_slices):
                
                axes[j, 0].imshow(volumes[i, 0, depth, :, :], cmap='gray')
                axes[j, 0].set_title(f'Input (Depth={depth})')
                axes[j, 0].axis('off')
                
                axes[j, 1].masks(volumes[i, 0, depth, :, :], cmap='gray')
                axes[j, 1].set_title(f'Mask')
                axes[j, 1].axis('off')
                
                axes[j, 2].imshow(preds[i, 0, depth, :, :], cmap='gray')
                axes[j, 2].set_title(f'Prediction')
                axes[j, 2].axis('off')

            plt.tight_layout()
            plt.show()
                
        
    

            