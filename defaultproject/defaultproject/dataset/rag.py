import os
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from utils.etc import hit2docdict
import gzip, pathlib, wget
import random
import pandas as pd


def unpack(gzip_file: str, out_file: str):
    input = gzip.GzipFile(gzip_file, "rb")
    s = input.read()
    input.close()
    output = open(out_file, "wb")
    output.write(s)
    output.close()


def donwload_dataset_nq_open_dpr(root_path='.'):
    # https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L30
    # data.wikipedia_split.psgs_w100
    URL_PASSAGE_W100 = 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz'
    # data.retriever.nq-dev
    URL_NQ_OPEN_DPR_DEV = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz'
    # data.retriever.nq-train
    URL_NQ_OPEN_DPR_TRAIN = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz'

    DIR_PASSAGE = os.path.join(root_path, 'data/wikipedia_split')
    DIR_NQ_OPEN_DPR = os.path.join(root_path, 'data/nq_open_dpr')

    FILE_NAME_PASSAGE = 'psgs_w100.tsv'
    FILE_NAME_NQ_DEV = 'nq_dev.json'
    FILE_NAME_NQ_TRAIN = 'nq_train.json'

    FULL_PATH_PASSAGE = os.path.join(DIR_PASSAGE, FILE_NAME_PASSAGE)
    FULL_PATH_NQ_DEV = os.path.join(DIR_NQ_OPEN_DPR, FILE_NAME_NQ_DEV)
    FULL_PATH_NQ_TRAIN = os.path.join(DIR_NQ_OPEN_DPR, FILE_NAME_NQ_TRAIN)
    # pathlib.Path(DIR_PASSAGE).mkdir(parents=True, exist_ok=True)
    # wget.download(URL_PASSAGE_W100, out=FULL_PATH_PASSAGE + '.gz')
    # unpack(FULL_PATH_PASSAGE + '.gz', FULL_PATH_PASSAGE)
    # os.remove(FULL_PATH_PASSAGE + '.gz')

    if not os.path.exists(FULL_PATH_NQ_DEV.replace('.json', '')):
        print(f"Dataset not found at '{FULL_PATH_NQ_DEV.replace('.json', '')}'. Downloading...")
        pathlib.Path(DIR_NQ_OPEN_DPR).mkdir(parents=True, exist_ok=True)
    
        print(f"wget download '{URL_NQ_OPEN_DPR_DEV}' to '{FULL_PATH_NQ_DEV}'")
        wget.download(URL_NQ_OPEN_DPR_DEV, out=FULL_PATH_NQ_DEV + '.gz')
        print(f"unpack '{FULL_PATH_NQ_DEV + '.gz'}' to '{FULL_PATH_NQ_DEV}'")
        unpack(FULL_PATH_NQ_DEV + '.gz', FULL_PATH_NQ_DEV)
        print(f"remove '{FULL_PATH_NQ_DEV + '.gz'}'")
        os.remove(FULL_PATH_NQ_DEV + '.gz')

        print(f"read json '{FULL_PATH_NQ_DEV}' to pandas")
        _ds_pd: pd.DataFrame = pd.read_json(FULL_PATH_NQ_DEV)
        print(f"convert pandas to huggingface Dataset")
        _ds_hfds = HF_Dataset.from_pandas(_ds_pd)
        print(f"save huggingface Dataset to '{FULL_PATH_NQ_DEV.replace('.json', '')}'")
        _ds_hfds.save_to_disk(FULL_PATH_NQ_DEV.replace('.json', ''))
        print(f"remove '{FULL_PATH_NQ_DEV}'")
        os.remove(FULL_PATH_NQ_DEV)
    else:
        print(f"Dataset already exists at '{FULL_PATH_NQ_DEV.replace('.json', '')}'")

    if not os.path.exists(FULL_PATH_NQ_TRAIN.replace('.json', '')):
        print(f"Dataset not found at '{FULL_PATH_NQ_TRAIN.replace('.json', '')}'. Downloading...")
        pathlib.Path(DIR_NQ_OPEN_DPR).mkdir(parents=True, exist_ok=True)

        print(f"wget download '{URL_NQ_OPEN_DPR_TRAIN}' to '{FULL_PATH_NQ_TRAIN}'")
        wget.download(URL_NQ_OPEN_DPR_TRAIN, out=FULL_PATH_NQ_TRAIN + '.gz')
        print(f"unpack '{FULL_PATH_NQ_TRAIN + '.gz'}' to '{FULL_PATH_NQ_TRAIN}'")
        unpack(FULL_PATH_NQ_TRAIN + '.gz', FULL_PATH_NQ_TRAIN)
        print(f"remove '{FULL_PATH_NQ_TRAIN + '.gz'}'")
        os.remove(FULL_PATH_NQ_TRAIN + '.gz')
        
        print(f"read json '{FULL_PATH_NQ_TRAIN}' to pandas")
        _ds_pd: pd.DataFrame = pd.read_json(FULL_PATH_NQ_TRAIN)
        print(f"convert pandas to huggingface Dataset")
        _ds_hfds = HF_Dataset.from_pandas(_ds_pd)
        print(f"save huggingface Dataset to '{FULL_PATH_NQ_TRAIN.replace('.json', '')}'")
        _ds_hfds.save_to_disk(FULL_PATH_NQ_TRAIN.replace('.json', ''))
        print(f"remove '{FULL_PATH_NQ_TRAIN}'")
        os.remove(FULL_PATH_NQ_TRAIN)
    else:
        print(f"Dataset already exists at '{FULL_PATH_NQ_TRAIN.replace('.json', '')}'")


class RAGDataset(Dataset):
    prefix_title = 'Title: '
    prefix_passage = 'Passage: '

    def __init__(
        self,
        tokenizer,
        is_train,
        dataset_path,
        num_samples=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.dataset = HF_Dataset.load_from_disk(dataset_path)

        if num_samples is not None:
            self.dataset = self.dataset.select(range(num_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs = self.dataset[index]
        if 'uid' not in inputs:
            inputs['uid'] = f"{'train' if self.is_train else 'eval'}_{index:04d}"
        uid: str = inputs['uid']
        question = inputs['question']
        answers = inputs['answers']
        answer = inputs['answers'][0]

        if self.is_train:
            positive_ctxs = inputs['positive_ctxs']
            ctx = random.choice(positive_ctxs)
            title = ctx['title']
            text = ctx['text']
            input_text_ctx = f"{self.prefix_title}{title} {self.prefix_passage}{text}"
            input_text_without_answer = f"{input_text_ctx}\nQuestion: {question}\nAnswer:"
        else:
            input_text_ctx=None
            input_text_without_answer=None
            
        if self.is_train:
            input_text = f"{input_text_without_answer} {answer}{self.tokenizer.eos_token}"
        else:
            input_text = None

        return {
                'input_text': input_text,
                'input_text_without_answer': input_text_without_answer,
                'question': question,
                'answer': answer,
                'input_text_ctx': input_text_ctx,
                'uid': uid,
                'answers': answers,
            }

class RAGCollator:
    def __init__(self, tokenizer, is_train=True) -> None:
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = 992

    def __call__(self, batch):
        list_input_text = []
        list_input_text_without_answer = []
        list_question = []
        list_answer = []
        list_input_text_ctx = []
        list_uid = []
        list_answers = []

        for samp in batch:
            list_input_text.append(samp['input_text'])
            list_input_text_without_answer.append(samp['input_text_without_answer'])
            list_question.append(samp['question'])
            list_answer.append(samp['answer'])
            list_input_text_ctx.append(samp['input_text_ctx'])
            list_uid.append(samp['uid'])
            list_answers.append(samp['answers'])

        if self.is_train:
            self.tokenizer.padding_side = "right"
            inputs_only_answer = self.tokenizer(
                [' '+answer+self.tokenizer.eos_token for answer in list_answer],
                padding="longest",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = "left"
            answer_length = inputs_only_answer['input_ids'].shape[1]
            inputs_without_answer = self.tokenizer(
                list_input_text_without_answer,
                padding="max_length",
                max_length=self.max_len-answer_length,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {}
            inputs['input_ids'] = torch.cat([inputs_without_answer['input_ids'], inputs_only_answer['input_ids']], dim=1)
            inputs['attention_mask'] = torch.cat([inputs_without_answer['attention_mask'], inputs_only_answer['attention_mask']], dim=1)
            
            inputs['labels'] = inputs['input_ids'].clone()
            inputs['labels'][:, :inputs_without_answer['input_ids'].shape[1]] = -100
            inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
            return inputs

        return {
            'inputs': None,
            'question': list_question,
            'answer': list_answer,
            'input_text_ctx': None,
            'uid': list_uid,
            'answers': list_answers,
        }
