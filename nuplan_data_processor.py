from Wayformer.wf_dataset import NuplanDataset
from Wayformer.utils import config

if __name__ == "__main__":
    args = config('wayformer.1')
    args.save_config('config')
    dataset = NuplanDataset(
        config=args,
        mode='train',
        to_screen=True,
        num_workers=4,
        reuse_cache=False,
        shuffle=True,
        val_ratio=0.2
    )
    val_dataset = NuplanDataset(
        config=args,
        mode='val',
        to_screen=True,
        num_workers=6,
        reuse_cache=False,
        shuffle=False,
        val_ratio=0.2
    )
    #initialize config file
    
for i in range(3):
    sample = dataset[i]
    print("future label shape:",sample['labels'].shape,"road matrix shape:",sample['matrix'].shape)