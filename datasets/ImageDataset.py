from monai.data import Dataset

class ImageDataset(Dataset):
  def __init__(self,img_path,mask_path,transform):
    self.data=[
        {'image':img, 'mask':mask}
        for img, mask in zip(img_path, mask_path)
    ]
    self.transform=transform
    self.length=len(self.data)

  def __len__(self):
    return self.length

  def __getitem__(self,idx):
    sample=self.transform(self.data[idx])
    return sample['image'], sample['mask']