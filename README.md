You might need to run this:
hf auth login           //USED TO LOG IN

python -m pip install --upgrade pip
pip install torch transformers pillow tqdm

.\.venv\Scripts\Activate.ps1

## Model Attribution
This project uses the [ShieldGemma 2-4B-IT](https://huggingface.co/google/shieldgemma-2-4b-it)
model by Google, released under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
Model weights are **not** redistributed; users must accept the Gemma Terms of Use to
download and run the model.

@misc{zeng2025shieldgemma2robusttractable,
    title={ShieldGemma 2: Robust and Tractable Image Content Moderation},
    author={Wenjun Zeng and Dana Kurniawan and Ryan Mullins and Yuchi Liu and Tamoghna Saha and Dirichi Ike-Njoku and Jindong Gu and Yiwen Song and Cai Xu and Jingjing Zhou and Aparna Joshi and Shravan Dheep and Mani Malek and Hamid Palangi and Joon Baek and Rick Pereira and Karthik Narasimhan},
    year={2025},
    eprint={2504.01081},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2504.01081},
}