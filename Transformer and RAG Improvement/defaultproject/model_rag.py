
from utils.etc import hit2docdict
import torch
# Modify
class ModelRAG():
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def search(self, queries, qids, k=5):

        # Use the retriever to get relevant documents
        list_passages = []
        list_scores = []

        # fill here
        ######
        
        for query in queries:
            hits = self.retriever.search(query, k) #상위 k 개 문서 반환
            passages = [hit2docdict(hit) for hit in hits]
            scores = [hit.score for hit in hits]

            list_passages.append(passages)
            list_scores.append(scores)
        
        ######

        return list_passages, list_scores

    # Modify
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        # Get the relevant documents for each query
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        # fill here
        ######
        
        for query, passages in zip(queries, list_passages):
            #검색된 문서들을 하나의 컨텍스트로 합치기
            context_parts = []
            for p in passages:
                context_parts.append(f"Title: {p['title']} Passage: {p['contents']}")
            context = "\n".join(context_parts)

            input_text = f"{context}\nQuestion: {query}\nAnswer:"
            list_input_text_without_answer.append(input_text)
        
        ######
        
        return list_input_text_without_answer

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # fill here:
        ######

        list_input_text = self.make_augmented_inputs_for_generate(queries, qids, k=k)

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            list_input_text,
            padding="longest",
            truncation=True,
            max_length=992,
            return_tensors="pt",
        )

        ######

        # # Move batch to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        
        outputs = outputs[:, inputs['input_ids'].size(1):]

        return outputs
