Popillia japonica ground truth: 20 labelled test views from Desktop/test_split_stacked (labelme polygons -> masks/, 255 = insect, rod excluded).
images/ are lossless PNG copies of the .tif. index.csv: category = pose index (p0..p12), specimen = s00..s33.
Run: python -m gtbench.run --data <this dir> --out <results dir> ...   (box prompt comes from the GT mask bbox)
