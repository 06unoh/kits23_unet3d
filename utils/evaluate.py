import torch

def evaluate(model, dataloader, criterion,dice_metric ,device):
    model.eval()
    avg_loss=0.0
    total=0
    dice_metric.reset()
    
    with torch.no_grad():
        for volumes, masks in dataloader:
            volumes, masks=volumes.to(device), masks.to(device)
            outputs=model(volumes)
            loss=criterion(outputs, masks)
            
            preds=torch.sigmoid(outputs)
            preds=(preds>=0.5).float()
            
            avg_loss+=loss.item()*volumes.size(0)
            total+=masks.size(0)            
            dice_metric(y_pred=preds, y=masks)
    avg_loss/=total
    dice_score=dice_metric.aggregate().item()
    return avg_loss, dice_score
            