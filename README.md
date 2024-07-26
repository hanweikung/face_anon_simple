# Face Anonymization Made Simple

## Setup

1. Clone the repository.

```bash
git clone https://github.com/hanweikung/fams.git
```

2. Create a Python environment by running the `environment.sh` script.

```bash
bash environment.sh
```

3. Download the models
    * [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
    * [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

## Anonymization
Modify the paths in the `infer_anon.sh` file as needed, and then run the script.

```bash
bash infer_anon.sh
```

We provide a sample dataset in the `my_dataset` directory, and the results of the anonymization inference will be saved in the `results` directory by default. If you wish to test the model on your own dataset, you can modify the file names in `my_dataset/metadata_anon.jsonl` and the URLs in `my_dataset/my_dataset_anon.py`.

## Face Swapping

Just like when running the inference script for anonymization, adjust the paths in the `infer_swap.sh` file, and run the script for face swapping inference.

```bash
bash infer_swap.sh
```

If you prefer to test the model with your own dataset, you can edit the file names in `my_dataset/metadata_swap.jsonl` and the URLs in `my_dataset/my_dataset_swap.py`.