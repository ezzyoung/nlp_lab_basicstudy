"""
Baseline RAG (PDF handout 요구사항)
- RAG + Few-shot prompting
- Llama-3.2-1B-Instruct 등 instruction-tuned LLM과 함께 사용
"""

from model_rag import ModelRAG


class BaselineRAG(ModelRAG):
    """
    PDF baseline RAG: RAG + Few-shot examples
    NQ train에서 예시를 선택해 프롬프트에 추가
    """

    def __init__(self):
        super().__init__()
        self.few_shot_examples = []

    def set_few_shot_examples(self, nq_train_dataset, n=3):
        """
        NQ train 데이터셋에서 n개 샘플을 Few-shot 예시로 설정
        """
        examples = []
        for i in range(n):
            sample = nq_train_dataset.dataset[i]
            ctx = sample["positive_ctxs"][0]
            title = ctx.get("title", "")
            text = ctx.get("text", ctx.get("contents", ""))
            context = f"Title: {title} Passage: {text}"
            examples.append({
                "question": sample["question"],
                "context": context,
                "answer": sample["answers"][0],
            })
        self.few_shot_examples = examples
        print(f"Few-shot examples {n}개 설정 완료")
        for i, ex in enumerate(examples):
            print(f"  {i+1}. Q: {ex['question'][:50]}...")
            print(f"     A: {ex['answer']}")

    def _build_few_shot_text(self):
        """Few-shot 예시 텍스트 구성"""
        few_shot_text = ""
        for ex in self.few_shot_examples:
            few_shot_text += (
                f"{ex['context']}\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ex['answer']}\n\n"
            )
        return few_shot_text

    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        """
        Few-shot 예시 + 검색 결과로 프롬프트 구성
        """
        few_shot_text = self._build_few_shot_text()
        list_passages, _ = self.search(queries, qids, k=k)

        list_input_text = []
        for query, passages in zip(queries, list_passages):
            context_parts = []
            for p in passages:
                context_parts.append(f"Title: {p['title']} Passage: {p.get('contents', p.get('text', ''))}")
            context = "\n".join(context_parts)
            input_text = f"{few_shot_text}{context}\nQuestion: {query}\nAnswer:"
            list_input_text.append(input_text)

        return list_input_text
