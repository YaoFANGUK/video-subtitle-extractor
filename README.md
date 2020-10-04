Source Code Hierarchy
==================
video-subtitle-extractor
├── checkpoints_mlt --  Tensorflow model trained by myself based on ICDAR 2017 MLT[1]
│   ├── checkpoint
│   ├── ctpn_50000.ckpt.data-00000-of-00001   -- model file
│   ├── ctpn_50000.ckpt.index
│   └── ctpn_50000.ckpt.meta
│
├── data  -- program output
│   ├── frames         -- output of the pre-processing module
│   ├── text_position  -- output of the CTPN module
│   ├── to_ocr         -- output of the CTPN module
│   └── to_srt
│        ├── to_srt.txt    -- raw output of OCR module
│        └── to_srt.srt/generated.srt  -- output of the post-processing module
│
├── ipynb      -- python notebook written from scratch      
│   ├── Accuracy Evaluation.ipynb      -- written from scratch
│   ├── GPU speed up.ipynb             -- written from scratch
│   ├── image localisation.ipynb       -- written from scratch
│   ├── OCR.ipynb                      -- written from scratch
│   ├── opencvsubtitles ver1.ipynb     -- written from scratch
│   ├── subtitle area detection.ipynb  -- written from scratch
│   └── subtitle generation.ipynb      -- written from scratch
│
├── main 			-- main code of the project
│   ├── accuracyCal.py     -- written from scratch
│   ├── demo.py            -- developed and built on eragonruan's[2] code
│   └── train.py           -- code adopted from eragonruan
│
├── model 			  -- model file trained by myself with daomin's [3] instructions
│   └── ocr           -- MXNet model based on MJSynth dataset[4] and SCD[5]
│
├── nets 			  -- Backbone Network of CNN
│   ├── model_train.py     -- code adopted from eragonruan
│   └── vgg.py             -- code adopted from eragonruan
│
└── nets 			  -- Third party code
    ├── bbox     
    ├── dataset    
    ├── prepare    
    └── text_connector         

Testing Dataset -- Dataset created and collected by myself
Testing Results -- Testing outputs and reports



1. install requirements before running
2. download trained model files from: (other Supplementary material also can be found here)
https://drive.google.com/drive/folders/1AuthnUK7bqYOlLcKi1GkoLyHbq3GyjfX?usp=sharing
3. unzip checkpoints_mlt.zip and put it into the checkpoints_mlt directory
4. demo video can be found at: https://youtu.be/0gq8FQHb448

## RUN:
```
python3 ./main/demo.py
```


## Performance Evaluation:
```
python3 ./main/accuracyCal.py
```
