Welcome to Default Final Project!

We'll be using PyTorch for this assignment. If you're not familiar with PyTorch, or if you would like to review some of the fundamentals of PyTorch, please check the some tutorials from pytorch web site, including:
https://colab.research.google.com/drive/13HGy3-uIIy1KD_WFhG4nVrxJC-3nUUkP?usp=sharing
https://www.youtube.com/watch?v=Uv0AIRr3ptg

To create your own Conda environment named CSE402 for the NLP course, please run the following command:

# 1. Create conda environment
 
     conda create -n cse402 python=3.12


# 2. Activate your environment:

    conda activate cse402


# 3. Install Jupyter notebook

    pip install notebook


# 4. Install pytorch, torchvision, torchaudio

     pip3 install torch torchvision torchaudio

     - This example uses Linux, pip, and CUDA 12.4. Please choose the installation option that matches your operating system and local settings.

# 5. Install additional packages (depending on your devices)

     pip3 install faiss-cpu pyserini wget transformers datasets "accelerate[torch]" evaluate absl-py nltk rouge_score


# To deactivate an active environment, use
    
    conda deactivate

## No `zip` command for Windows
If you are using Windows, you may see an error when running the `collect_submission.sh` script. This is because the `zip` command is not available by default on Windows. 
If you encounter this error, you can try other methods mentioned [here](https://superuser.com/questions/201371/create-zip-folder-from-the-command-line-windows). Alternatively, you can manually zip the Python files.


"faiss-cpu",
        "pyserini",
        "wget",
        "transformers",
        "datasets",
        "accelerate[torch]",
        "evaluate",
        "absl-py",
        "nltk",
        "rouge_score"