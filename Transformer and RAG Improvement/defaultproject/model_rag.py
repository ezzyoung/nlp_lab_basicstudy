
from utils.etc import hit2docdict
import torch
# Modify
'''
main.py 에서 이렇게 불러옴

model_rag = ModelRAG()

model_rag.set_model(model)        # GPT-small or Llama 모델
model_rag.set_retriever(retriever) # BM25 검색기
model_rag.set_tokenizer(tokenizer) # 토크나이저


역색인 만드는 과정은 Pyserini 가 이미 만들어 서버에 올림.
우린 그걸 사용해서 인덱스 다운로드 하고 디스크에 저장 과정에서 인덱스도 저장
Lucene 내부에서 BM25 점수 계산함 그리고 임베딩 안쓴거 BM25 는 임베딩 없음

BM25 인덱스 구조
그냥 단어 → 문서번호 대응표

"goblin"    → [12, 901, 3421, 8823, ...]
"king"      → [12, 55,  901,  2341, ...]
"labyrinth" → [12, 788, 9021, ...]
"paris"     → [99, 234, 5521, ...]
...
'''
class ModelRAG():
    def __init__(self):
        pass
    #모델 바꿔낄 수 있게 유연하게 설계
    def set_model(self, model):
        self.model = model # 나중에 generate() 호출시 gpt or llama

    def set_retriever(self, retriever):
        self.retriever = retriever #.search() 메서드로 위키피디아 검색 bm25 검색기

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer #.encode(), decode() 씀 -> 토큰화, 디코딩

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
      
#quids 는 쿼리의 id 로 질문 한 문장에 대한 번호이고 Pytorch Dataloader 가 자동 생성
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
            context = "\n".join(context_parts) #검색한 topk=5 문서를 하나의 context 로 합침

            input_text = f"{context}\nQuestion: {query}\nAnswer:"
            list_input_text_without_answer.append(input_text)
        
        ######
        
        return list_input_text_without_answer #topk=5 를 받아서 질문 쿼리에 대한 검색 문서를 컨텍스트로 함께 제공받아 정확한 답변 생성하게 만들기

    @torch.no_grad() #이 함수 안에서는 gradient 계산 안함 지정
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # fill here:
        ######

        list_input_text = self.make_augmented_inputs_for_generate(queries, qids, k=k) #이걸 인풋 넣는다

        #토큰화 시킴
        self.tokenizer.padding_side = "left" #패딩을 왼쪽으로 맞춰야 Answer: 다음에 답변 생성됨
        inputs = self.tokenizer(
            list_input_text, #프롬프트 문자열 리스트
            padding="longest", #가장 긴 시퀀스 기준으로 맞춘다 패딩
            truncation=True, #max_length 넘어가면 자르기
            max_length=992, #992 토큰으로 제한
            return_tensors="pt", #파이토치 텐서로 반환
        )

        ######
        # 마스킹의 경우 패딩 영역만 마스크, 나머지는 전부 1
        #  
        # # Move batch to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()} #cpu -> gpu 를 텐서 저장된 v 를 올림, 딕셔너리 만들기 위해 사용
        outputs = self.model.generate(
            **inputs,
            **kwargs
        ) #**kwargs 는 지정되지 않은 것들 중 함수 호출단계에서 지정된 파라미터가 없을때 kwargs 로 넘겨짐 딕셔너리 풀리면서 키가 파라미터 이름이 됨
        #** 지정하면 변수의 딕셔너리가 풀려서 a=b 형태로 지정되어 들어감
        outputs = outputs[:, inputs['input_ids'].size(1):] #inputs['input_ids'].size(1) inputs['input_ids'] 는 (batch, seq_len) 이고 .size(1) 은 인덱스 1인 seq_len 길이를 뽑겠다는 말 즉 이거 이후만 truncate 된 답만을 출력하겠다는 뜻
        #output shape 은 (batch, 프롬프트 길이 + 생성 길이)

        return outputs #토큰 id 텐서로 출력

        #unsqueeze(-1) -> 마지막 차원에 차원 추가. 이런식으로 특정 곳에 차원 추가
        #next_token 에는 배치 내 각 샘플의 다음 토큰 id 가 존재. 즉 [batch_size] 만 있는 상태 그렇기 때문에 unsqueeze(-1) 로
        #seq_len 방향 1칸 추가시켜야 함
        #dim=-1 : 이 방향으로 시퀀스 늘려라 torch.cat([input_ids, new_token], dim=-1)
