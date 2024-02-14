# PyTorch-Image-Dehazing
PyTorch implementation of some single image dehazing networks.


Currently Implemented: An extremely lightweight model (< 10 KB). Results are good.



**Prerequisites:**

1.Python 3.7

2.Pytorch 1.7.0



**Training:**

1. Run train.py. The script will automatically dump some validation results into the "samples" folder after every epoch. The model snapshots are dumped in the "snapshots" folder. 



**Testing:**

1.Run dehaze.py. The script takes images in the "testdata" folder and dumps the dehazed images into the "result_my" folder. A pre-trained snapshot has been provided in the snapshots folder.


**Folder description:**

1.tesedata:Image of the test set to be enhanced

2.result_my:Test set images enhanced with our method

3.test_data:The real clear picture of the test set(clean), the image to be enhanced(hazy), the image enhanced by our method(result_my), and the image enhanced by AODNet(result_AODNet)
