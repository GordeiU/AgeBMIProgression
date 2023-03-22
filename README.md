# AgeBMIProgression

The project is the part of the final-year honors project

# Run Book

1. Unzip the `data.zip`, it should have the following structure:

```
   data
   |- unlabeled
       |-A00147_3_1.jpg
       ...
```

2. Create conda environment from the `environment_droplet.yml` file by running `conda <NAME_OF_ENVIRONMENt> create -f environment.yml`
3. Run `python main.py --mode train --execution cpu --debug` change the `--execution cuda` if you have a cuda enabled GPUs and use. The first time you run it the it will create `labeled` folder under the data with the subfolder of labels

Note: the full command is:
```
main.py [-h] [--mode {train,test}] [--epochs EPOCHS] [--models-saving {always,last,tail,never}] [--batch-size BATCH_SIZE] [--weight-decay WEIGHT_DECAY]
               [--learning-rate LEARNING_RATE] [--b1 B1] [--b2 B2] [--shouldplot SP] [--age AGE] [--bmi BMI] [--bmi_Group BMI_GROUP] [--watermark] [--execution {mps,cpu,cuda}][--load LOAD] [--input INPUT] [--output OUTPUT] [--debug] [--no-debug] [-z Z_CHANNELS]
```

# Acknowledgments

[Original Age Progression GitHub](https://github.com/mattans/AgeProgression)