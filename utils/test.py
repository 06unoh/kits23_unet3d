import torch

def test(model, dataloader ,dice_metric ,device):
    model.eval()
    dice_metric.reset()
    
    with torch.no_grad():
        for volumes, masks in dataloader:
            volumes, masks=volumes.to(device), masks.to(device)
            outputs=model(volumes)
            
            preds=torch.sigmoid(outputs)
            preds=(preds>=0.5).float()
            
            dice_metric(y_pred=preds, y=masks)
    
    dice_score=dice_metric.aggregate().item()
    print(f'Dice Score : {dice_score*100:.2f}%')
            