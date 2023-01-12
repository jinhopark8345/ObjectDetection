This is a practice repo to convert [blog-post](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f), [notebook](https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7) to a ML project.
 

### How to setup
Device : NVIDIA 3080ti 

```bash
conda create -n efficientdet python==3.9
conda activate efficientdet

# make sure to install correct torch version that matches with your device
# current requirements.txt is for 3080ti
pip install -r requirements.txt
```

### single run
```bash
# train & inference
python3 notebook/single_run.py
```

### train
```bash
python3 trainer/trainer.py
```

### test
```bash
python3 run/inference.py

```

### Reference
- blog:  
- notebook: )
