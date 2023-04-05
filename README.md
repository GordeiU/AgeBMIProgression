# AgeBMIProgression

The project is the part of the final-year honors project

# Virtual Environment

```shell
conda create -n ageProgression python=3.7 anaconda
```

```shell
conda activate ageProgression
```

```shell
pip install -r requirements.txt
```

# Training

```shell
python main.py --mode train --epochs NUMBER --output checkpoints
```

# Run Book Data

1. Unzip the `data.zip`, it should have the following structure:

```
   data
   |- unlabeled
       |-A00147_3_1.jpg
       ...
```

# Acknowledgments

[Original Age Progression GitHub](https://github.com/mattans/AgeProgression)
