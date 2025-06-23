from monai import transforms as mt

def get_train_tf():
    return mt.Compose([
        mt.LoadImaged(keys=['image', 'mask'], reader='NibabelReader'),
        mt.EnsureChannelFirstd(keys=['image','mask']),
        mt.Resized(keys=['image', 'mask'], spatial_size=(128, 128, 128)),
        mt.ScaleIntensityRanged(
            keys=['image'], a_min=-100, a_max=400, b_min=0.0, b_max=1.0
        ),
        mt.RandFlipd(keys=['image','mask'], prob=0.5, spatial_axis=[0,1,2]),
        mt.RandAffined(
            keys=['image', 'mask'],
            mode=['bilinear', 'nearest'],
            prob=0.5,
            rotate_range=(0.1,0.1,0.1),
            scale_range=(0.8,1.2)
        ),
        mt.Rand3DElasticd(
            keys=['image','mask'],
            sigma_range=(2,4),
            magnitude_range=(5,10),
            prob=0.5,
            mode=('bilinear','nearest')
        ),
        mt.RandCoarseDropout(holes=10, spatial_size=20, prob=0.5),
        mt.NormalizeIntensityd(keys=['image'], nonzero=True)
    ])

    
def get_test_tf():
    return mt.Compose([
        mt.LoadImaged(keys=['image','mask'],reader='NibabelReader'),
        mt.EnsureChannelFirstd(keys=['image','mask']),
        mt.Resized(keys=['image', 'mask'], spatial_size=(128, 128, 128)),
        mt.ScaleIntensityRanged(
            keys=['image'], a_min=-100, a_max=400, b_min=0.0, b_max=1.0
        ),
        mt.NormalizeIntensityd(keys=['image'],nonzero=True)
    ])