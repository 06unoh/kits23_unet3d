
    
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    avg_loss=0.0
    total=0
    
    for volumes, masks in dataloader:
        volumes, masks=volumes.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs=model(volumes)
        loss=criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        avg_loss+=loss.item()*volumes.size(0)
        total+=masks.size(0)
        
    avg_loss/=total
    return avg_loss