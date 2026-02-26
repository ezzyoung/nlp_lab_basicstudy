import json
import random
import numpy as np
import torch
from pyserini.index.lucene import Document


def print_model_statistics(model):
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    n_params_trainable = sum({p.data_ptr(): p.numel() for p in model.parameters() if p.requires_grad}.values())
    print(f"Training Total size={n_params / 2**20:.2f}M params. Trainable ratio={n_params_trainable / n_params * 100:.2f}%")

#Pyserini 는 BM25 를 파이썬에서 쓸 수 있게 하는 라이브러리 -> Lucene (JAVA 검색 엔진) 사용
def set_seed_everything(seed=42): #seed 고정해서 동일 결과 및 재현성 보장
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hit2docdict(hit) -> dict:
    return json.loads(Document(hit.lucene_document).raw())

#위키피디아 passage -> pyserini 가 인덱싱 -> lucene 인덱스 (java 형식으로 디스크 저장) -> 검색 했을때 hit 나오면 hit.lucene_document 에 저장된 문서 내용 가져오기
#hits = [hit1...hit5] -> k=5 개 문서 반환
