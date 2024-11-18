# Transformer for English-to-Chinese Translation

This repository implements a seminal transformer design inspired by the principles outlined in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). The coding implementation was adopted from a [tutorial](https://www.youtube.com/watch?v=ISNdQcPhsts) made by Umar Jamil. More details can be found at `Project Report.pdf`.

The model is trained on the English-Chinese section of the [OPUS-100 dataset](https://huggingface.co/datasets/Helsinki-NLP/opus-100) on Hugging Face. 

- **Training Data**: 1 million samples  
- **Validation Data**: 2,000 samples  
- **Test Data**: 2,000 samples


# Installation

## Requirements

- __Hardware__: The model is trained and evaluated on an NVIDIA 3070 GPU. While hardware with lower specifications may be compatible, it could result in longer training time.

- __System__: Ubuntu 22.04

## Dependency Installation

- __Anaconda__: download and install [Anaconda](https://www.anaconda.com/download/success).

## Package Installation
Create a new Conda environment named `transformer` and install all necessary packages within this environment.

```bash
conda create -n transformer python=3.10
conda activate transformer
```

Clone the repository from GitHub.
```bash
git clone https://github.com/QZJGeorge/STATS507-Final-Project.git
```

Install the required python packages
```
pip install -r requirements.txt
```

# Model Training

For detailed training parameters, refer to `config.py`. To start training, run the following command. This will train and save the transformer model.

```bash
python3 train.py
```

You can also download our [pre-trained model](https://drive.google.com/file/d/1pgEhp-gVIXI6nkYK7IPYwJXsyrHZU06T/view?usp=sharing) and place it in the Helsinki-NLP/opus-100_weights folder.

# Model Evaluation

The validation and test datasets were combined, resulting in a total of 4,000 samples. The model's performance is evaluated using [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore). 

To run the evaluation, execute the following command:

```bash
python3 evaluate.py
```

Our pre-trained model achieved a Precision score of 0.795, Recall score of 0.766, and F1 score of 0.779.

# Troubleshooting

If you encounter the error `torch.cuda.OutOfMemoryError`, open `config.py` and reduce the batch size.

# Developer

Zhijie Qiao zhijieq@umich.edu

# License

Distributed under the MIT License.




