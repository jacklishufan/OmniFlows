### Install additional packages

```
pip install -r requirements_train.txt
```

### Dataset Preparation 

For paired multi-modal data, you need to prepare a csv file for each dataset listing the full path of media. 

For image-text dataset, the format is as follows

```
caption,img_file
an image of dog,/abs/path/to/img.jpg
...
```

For image-audio dataset, the format is as follows

```
img_path,audio_path
/abs/path/to/img.jpg,/abs/path/to/aud.mp3
...
```

For text-audio dataset, the format is as follows

```
caption,audio_path
Music is playing and a frog croaks,/abs/path/to/aud.mp3
...
```


For text-image-audio dataset, the format is as follows

```
caption2,caption,img_path,audio_path
A person's face contorted in a howling expression,A person makes a howling sound,/abs/path/to/img.jpg,/abs/path/to/aud.mp3
...
```

`caption2` column contains visual captions,`caption` column contains audio captions. For dataset with only one caption (e.g. Video dataset), we can repeat the same caption in two columns
  /data0/jacklishufan/OmniFlow-public/scripts/training.md
Finally, you need to prepare a dataset config which contains all the csv files of different dataset and the sampling weight of each dataset during the training. An example can be found at [config/data.yml](../config/data.yml)

### Launch Training

We provide an example launch script [scripts/example.sh](scripts/example.sh)
