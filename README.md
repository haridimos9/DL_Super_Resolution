# DL_Super_Resolution

This repository contains 3 experiments used by our team for the task of Super Resolution on [Veo's](https://www.veo.co/da) footbal matches.
* ESRGAN (Enhanced SRGAN) Single Image Super Resolution) 
* SRGAN 3 (Multi Image Super Resolution)
* iSeeBetter (Video Super Resolution)

In each folder there is the code for each experiment we conducted.
Additionally, here lies the [explainer notebook](https://github.com/haridimos9/DL_Super_Resolution/blob/main/reproduce.ipynb) where you can test the results on one (or more) images.

Additionally, you can have a sneak peak of the results from the demo gif, where we compare an original clip, with the one created from the ESRGAN model, one created from iSeeBetter with MSE loss, and one with the paper's APIT loss:
![Alt Text](https://github.com/haridimos9/DL_Super_Resolution/blob/main/comparison.gif)

## Installation

Clone the repository and then navigate in the folder of the model you want to test, eg. ESRGAN: `cd ESRGAN`. Afterwards, create a virtual environment and install the requiremtns for the project. Here `python3-venv` is used:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Create a folder to store the test/inference images in `png` format. Otherwise, you can store the images/frames in an existing folder, eg. `data`.
In the terminal write the following command:
```bash
python implementations/ESRGAN/test_on_many_images.py --help
```
to see al the required arguments for the file `test_on_many_images.py`. An example run can be:
```bash
python implementations/ESRGAN/test_on_many_images.py --images_path data --checkpoint_model weights
```
where the frames are stores in the directory `data` and the model checkpoints, in `.pth` format, are saved in the directory `weights`, at the same level as `implementations`. For the ESRGAN model the trained weights can be fetced from
```
https://drive.google.com/file/d/1qrwUmyzggJc-NbwnfADH4MyXlHEDx2Ua/view?usp=sharing
```

#### Note about the `iSeeBetter` model
For the `iSeeBetter` model, the consecutive frames must be stored in folders named `1`, `2`, inside the main data directory, eg. `data` or `Vid4`. Inide a frames folder, the frames must be named `im1.png` up to `im7.png`.

Moreover, a `.txt` file is created and stored in `data`, where the frames to use are stored. Example are located in the [Vid4](https://github.com/haridimos9/DL_Super_Resolution/tree/main/iSeeBetter/Vid4) folder.

After the frames have been stored alongside the `frames_to_load.txt` file, the test can be run using the following command, under a virtual environment:
```bash
python iSeeBetterTest.py --data_dir Vid4 --file_list frames_to_load.txt 
```
where the frames are stored in the directory Vid4 with the specified format.
If the APIT Loss model is required to be used for testing:

```bash
python iSeeBetterTest.py --data_dir Vid4 --file_list frames_to_load.txt --model  netG_epoch_4_1_APITLoss.pth 
```

More arguments can be seen by the command
```bash
python iSeeBetterTest.py --help
```

## Results reproduction
A quick test can be run using the file `reproduce.ipynb`. After having installed all the required libraries (requirements.txt) for each model, open the jupyter notebook `reproduce.ipynb` and run all the cells.
