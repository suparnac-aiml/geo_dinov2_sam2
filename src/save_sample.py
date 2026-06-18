from datasets import load_dataset

ds = load_dataset('blanchon/LEVIR_CDPlus', split='test')
ds[0]['image1'].save('/tmp/sample_t1.png')
ds[0]['image2'].save('/tmp/sample_t2.png')
print('Saved sample images to /tmp/')
