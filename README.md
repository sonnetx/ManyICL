# BiasICL: Many-Shot In-Context Learning in Multimodal Foundation Models


This repository contains implementation of [ManyICL](https://arxiv.org/abs/2405.09798). Prepare a dataframe, configure your API key, modify the prompt and just run it!

Please note that this code repo is intended for research purpose, and might not be suitable for large-scale production.


# Installation
Install packages using pip:
```bash
$ pip install -r requirements.txt
```

# Setup API keys
## For GPT-series models offered by OpenAI
1. Get your API key from [here](https://platform.openai.com/api-keys);
2. Replace the placeholder in `ManyICL/LMM.py` (Line 2);

## For Gemini-series models
Note that you need a Google API key for this. 

# Dataset preparation
Prepare two pandas dataframe: one for the demonstration set and one for the test set. You can find examples under the `dataset/` folder. Note that the index column should contain the filenames of the images. Here's a quick preview: 

| Index | Forest | Golf course | Freeway |
|:-------------|:--------------:|:--------------:|:--------------:|
|forest39.jpeg| 1 | 0 | 0 |
|golfcourse53.jpeg| 0 | 1 | 0 |
|freeway97.jpeg| 0 | 0 | 1 |

## Expected directory structure
Note that we only include 42 images in UCMerced dataset for illustration purposes. 

```
ManyICL/
├── LMM.py
├── dataset
│   └── UCMerced
│       ├── demo.csv
│       ├── test.csv
│       ├── images
│       │   ├── forest39.jpeg
│       │   ├── forest47.jpeg
│       │   ├── freeway09.jpeg
│       │   ├── freeway97.jpeg
│       │   ├── ...
├── prompt.py
└── run.py

```

# Configure the prompt

Modify the prompt in `prompt.py` if needed.

# Run the experiment
Run the experiment script, and it'll save all the raw responses in `UCMerced_21shot_Gemini1.5_7.pkl`.
```bash
python3 ManyICL/run.py --dataset=UCMerced --num_shot_per_class=1 --num_qns_per_round=7
```

# Evaluate the model responses
Run the evaluation script, and it'll read from the raw responses and print out the accuracy score.
```bash
python3 ManyICL/eval.py --dataset=UCMerced --num_shot_per_class=1 --num_qns_per_round=7
```

# Citation

If you find our work useful in your research please consider citing:

```
@misc{jiang2024manyshot,
      title={Many-Shot In-Context Learning in Multimodal Foundation Models}, 
      author={Yixing Jiang and Jeremy Irvin and Ji Hun Wang and Muhammad Ahmed Chaudhry and Jonathan H. Chen and Andrew Y. Ng},
      year={2024},
      eprint={2405.09798},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
