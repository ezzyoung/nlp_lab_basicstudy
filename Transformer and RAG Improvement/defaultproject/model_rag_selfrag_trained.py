"""
Self-RAG (논문 버전 정렬)

Asai et al. 2024 "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
- 적응형 검색: [Retrieval] vs [No Retrieval] 모델이 선택
- 생성 중 reflection: [Relevant], [Fully supported], [Utility:X] 등 출력
- [Generate] 토큰: 답변 생성 전 사용
- 반복적 검색: 생성 중 [Retrieval] 재출력 시 추가 검색

과제(NQ) 사용법:
  1. NQ 로드: nq_train = load_from_disk(".../nq_train"), nq_dev = load_from_disk(".../nq_dev")
  2. add_selfrag_tokens(tokenizer, model)
  3. Critic 라벨: critic_for_labels=CriticModel(pretrained_gpt, tokenizer) 또는 None(휴리스틱)
  4. CriticDataset(tokenizer, nq_train, retriever, critic_for_labels=...) → train_critic()
  5. CriticModel(trained_critic, tokenizer) → SelfRAGDataset(critic=...) (critic 필수)
  6. train_selfrag() → Generator 학습 (NQ)
  7. ModelRAGTrainedSelfRAG.retrieval_augmented_generate()

평가 시: ModelRAGTrainedSelfRAG 사용 시 eval_for_rag에서
  pred = model.decode_outputs_for_eval(outputs)
  로 답변만 추출하여 사용 (skip_special_tokens=True 대신)
"""

from model_rag import ModelRAG
from utils.etc import hit2docdict
import torch
import re
from typing import List, Optional, Tuple, Callable
from datasets import Dataset as HF_Dataset


# ── 논문 Self-RAG 특수 토큰 (공식 형식) ────────────────────────────────────

SELF_RAG_TOKENS = [
    "[Retrieval]", "[No Retrieval]", "[Generate]",
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]

# 논문 프롬프트 형식
INSTRUCTION_TEMPLATE = "### Instruction:\n{question}\n\n### Response:\n"
CONTEXT_TEMPLATE = "[Retrieval]{context}"


def add_selfrag_tokens(tokenizer, model):
    """토크나이저에 Self-RAG 특수 토큰 추가, 모델 임베딩 확장"""
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": SELF_RAG_TOKENS
    })
    if num_added > 0 and model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return num_added


def _build_context(passages: List[dict]) -> str:
    """passage 리스트 → 컨텍스트 문자열 (ModelRAG 형식과 호환)"""
    parts = []
    for p in passages:
        title = p.get("title", "")
        text = p.get("text", p.get("contents", ""))
        parts.append(f"Title: {title} Passage: {text}")
    return "\n".join(parts)


# ── Critic (논문 정렬) ──────────────────────────────────────────────────────

class CriticModel:
    """
    논문 Critic 역할: 검색 문서 관련성, 답변 지원 여부, 품질 평가
    - 논문: GPT-4 또는 학습된 Critic 모델로 라벨 생성
    - 본 구현: 동일 모델에 프롬프트로 평가 요청 (Critic 시뮬레이션)
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def set_model(self, model):
        self.model = model

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _generate_choice(self, prompt: str, choices: List, max_new_tokens: int = 8) -> str:
        """프롬프트에 대해 choices 중 하나 생성 (학습된 Critic은 특수 토큰 직접 출력)"""
        if self.model is None or self.tokenizer is None:
            return choices[0]
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=False,
        ).strip()
        response_lower = response.lower()
        for c in choices:
            if isinstance(c, str):
                if c in response or c.lower() in response_lower:
                    return c
            if isinstance(c, int) and str(c) in response:
                return c
        return choices[0]

    def is_relevant(self, question: str, passage: dict) -> str:
        """[Relevant] | [Irrelevant]"""
        ctx = _build_context([passage])
        prompt = f"{ctx}\nQuestion: {question}\nIs this passage relevant to answer the question? Answer: relevant or irrelevant:"
        return "[Relevant]" if self._generate_choice(prompt, ["relevant", "irrelevant"]) == "relevant" else "[Irrelevant]"

    def is_supported(self, question: str, answer: str, passages: List[dict]) -> str:
        """[Fully supported] | [Partially supported] | [No support]"""
        if not passages:
            return "[No support]"
        ctx = _build_context(passages)
        prompt = f"{ctx}\nQuestion: {question}\nAnswer: {answer}\nIs this answer supported by the passages? Answer: fully supported, partially supported, or no support:"
        choice = self._generate_choice(prompt, ["fully supported", "partially supported", "no support"])
        mapping = {"fully supported": "[Fully supported]", "partially supported": "[Partially supported]", "no support": "[No support]"}
        return mapping.get(choice, "[No support]")

    def utility(self, question: str, answer: str) -> str:
        """[Utility:1] ~ [Utility:5]"""
        prompt = f"Question: {question}\nAnswer: {answer}\nRate 1-5 how well this answers the question (1=poor, 5=excellent). Answer with one number:"
        result = self._generate_choice(prompt, [5, 4, 3, 2, 1])
        return f"[Utility:{result}]" if isinstance(result, int) else "[Utility:3]"


# ── 휴리스틱 Critic (Critic 모델 없을 때 fallback) ───────────────────────────

def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _is_answer_in_passage(answer: str, passage_text: str) -> bool:
    na = _normalize(answer)
    np = _normalize(passage_text)
    return na in np or any(_normalize(a) in np for a in na.split() if len(a) > 2)


def _question_passage_overlap(question: str, passage_text: str) -> float:
    qw = set(_normalize(question).split())
    pw = set(_normalize(passage_text).split())
    return len(qw & pw) / len(qw) if qw else 0.0


def get_critic_labels_heuristic(
    question: str,
    answer: str,
    passages: List[dict],
    gold_passage: Optional[dict] = None,
) -> Tuple[List[str], str, str]:
    """
    휴리스틱 Critic 라벨 (논문 토큰 형식)
    Returns: rel_labels, sup_label, use_label
    """
    rel_labels = []
    for p in passages:
        text = p.get("text", p.get("contents", ""))
        is_gold = gold_passage and text == gold_passage.get("text", gold_passage.get("contents", ""))
        overlap = _question_passage_overlap(question, text)
        rel_labels.append("[Relevant]" if is_gold or overlap > 0.2 else "[Irrelevant]")

    has_support = False
    partial = False
    if gold_passage:
        gt = gold_passage.get("text", gold_passage.get("contents", ""))
        if _is_answer_in_passage(answer, gt):
            has_support = True
        else:
            partial = any(_is_answer_in_passage(answer, p.get("text", p.get("contents", ""))) for p in passages)
    else:
        for p in passages:
            pt = p.get("text", p.get("contents", ""))
            if _is_answer_in_passage(answer, pt):
                has_support = True
                break
            if any(w in _normalize(pt) for w in _normalize(answer).split() if len(w) > 3):
                partial = True

    if has_support:
        sup_label, use_label = "[Fully supported]", "[Utility:5]"
    elif partial:
        sup_label, use_label = "[Partially supported]", "[Utility:4]"
    else:
        sup_label, use_label = "[No support]", "[Utility:2]"

    return rel_labels, sup_label, use_label


# ── Critic 학습 (논문 Step 1~2) ──────────────────────────────────────────────

class CriticDataset(torch.utils.data.Dataset):
    """
    Critic 학습 데이터: (prompt, label_token) 쌍
    과제: NQ 데이터 사용
    라벨: critic_for_labels(GPT 모델) 사용 시 해당 모델로 생성, 없으면 휴리스틱 (논문은 GPT-4)
    """

    def __init__(
        self,
        tokenizer,
        hf_dataset: HF_Dataset,
        retriever,
        critic_for_labels: Optional[CriticModel] = None,
        max_length: int = 512,
        num_passages: int = 5,
        num_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.dataset = hf_dataset
        self.retriever = retriever
        self.critic_for_labels = critic_for_labels
        self.max_length = max_length
        self.num_passages = num_passages

        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))

        self._samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            question = sample["question"]
            answer = sample["answers"][0]
            gold_ctx = sample.get("positive_ctxs", [{}])[0] if sample.get("positive_ctxs") else None

            hits = self.retriever.search(question, self.num_passages)
            passages = [hit2docdict(h) for h in hits]

            if self.critic_for_labels is not None:
                rel_labels = [self.critic_for_labels.is_relevant(question, p) for p in passages]
                sup_label = self.critic_for_labels.is_supported(question, answer, passages)
                use_label = self.critic_for_labels.utility(question, answer)
            else:
                rel_labels, sup_label, use_label = get_critic_labels_heuristic(
                    question, answer, passages, gold_ctx
                )

            for p, rel in zip(passages, rel_labels):
                ctx = _build_context([p])
                prompt = f"Context: {ctx}\nQuestion: {question}\nIs this passage relevant? Answer:"
                samples.append({"prompt": prompt, "label": rel})

            ctx_all = _build_context(passages)
            prompt_sup = f"Context: {ctx_all}\nQuestion: {question}\nAnswer: {answer}\nIs this answer supported? Answer:"
            samples.append({"prompt": prompt_sup, "label": sup_label})

            prompt_use = f"Question: {question}\nAnswer: {answer}\nRate 1-5. Answer:"
            samples.append({"prompt": prompt_use, "label": use_label})

        return samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        full_text = s["prompt"] + " " + s["label"] + self.tokenizer.eos_token
        return {"text": full_text, "prompt_len": len(self.tokenizer.encode(s["prompt"], add_special_tokens=True))}


class CriticCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [s["text"] for s in batch]
        prompt_lens = [s["prompt_len"] for s in batch]
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_critic(
    model,
    tokenizer,
    train_dataset: CriticDataset,
    train_config: dict,
):
    """
    논문 Step 2: Critic 모델 학습
    휴리스틱 라벨로 Critic이 [Relevant], [Fully supported], [Utility:5] 등 예측하도록 학습
    """
    from torch.utils.data import DataLoader

    add_selfrag_tokens(tokenizer, model)
    collator = CriticCollator(tokenizer, max_length=train_config.get("max_length", 512))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=True,
        collate_fn=collator,
    )

    device = train_config.get("device", "cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 5e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
    )

    model.train()
    for epoch in range(train_config.get("num_epochs", 2)):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


# ── Self-RAG 학습 데이터셋 ──────────────────────────────────────────────────

class SelfRAGDataset(torch.utils.data.Dataset):
    """
    논문 형식 Self-RAG Generator 학습 시퀀스
    과제: NQ 데이터 사용. critic 필수 (학습된 Critic 또는 만든 GPT 모델)
    - [Retrieval] 경로: ### Instruction + [Retrieval]{context}[Relevant]...[Generate] answer [Fully supported][Utility:5]
    - [No Retrieval] 경로: ### Instruction + [No Retrieval][Generate] answer [Fully supported][Utility:5]
    """

    def __init__(
        self,
        tokenizer,
        hf_dataset: HF_Dataset,
        retriever,
        critic: CriticModel,
        max_context_len: int = 800,
        num_passages: int = 5,
        num_samples: Optional[int] = None,
        no_retrieval_ratio: float = 0.2,
    ):
        if critic is None:
            raise ValueError("critic is required. Use CriticModel(trained_model, tokenizer) or CriticModel(pretrained_gpt, tokenizer)")
        self.tokenizer = tokenizer
        self.dataset = hf_dataset
        self.retriever = retriever
        self.critic = critic
        self.max_context_len = max_context_len
        self.num_passages = num_passages
        self.no_retrieval_ratio = no_retrieval_ratio

        if num_samples is not None:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import random
        sample = self.dataset[idx]
        question = sample["question"]
        answer = sample["answers"][0]
        gold_ctx = sample.get("positive_ctxs", [{}])[0] if sample.get("positive_ctxs") else None

        hits = self.retriever.search(question, self.num_passages)
        passages = [hit2docdict(h) for h in hits]

        rel_labels = [self.critic.is_relevant(question, p) for p in passages]
        sup_label = self.critic.is_supported(question, answer, passages)
        use_label = self.critic.utility(question, answer)

        all_irrelevant = all(r == "[Irrelevant]" for r in rel_labels)
        use_no_retrieval = all_irrelevant or (random.random() < self.no_retrieval_ratio)

        if use_no_retrieval:
            parts = [INSTRUCTION_TEMPLATE.format(question=question), "[No Retrieval]", "[Generate]"]
            full_text = f"\n".join(parts) + f" {answer} {sup_label} {use_label}{self.tokenizer.eos_token}"
        else:
            parts = [INSTRUCTION_TEMPLATE.format(question=question), "[Retrieval]"]
            for p, rel in zip(passages, rel_labels):
                text = p.get("text", p.get("contents", ""))
                title = p.get("title", "")
                parts.append(f"Title: {title} Passage: {text}")
                parts.append(rel)
            parts.append("[Generate]")
            context_str = "\n".join(parts)
            if len(context_str) > self.max_context_len * 4:
                context_str = context_str[: self.max_context_len * 4]
            full_text = f"{context_str} {answer} {sup_label} {use_label}{self.tokenizer.eos_token}"

        return {"text": full_text, "question": question, "answer": answer}


class SelfRAGCollator:
    def __init__(self, tokenizer, max_length: int = 992):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [s["text"] for s in batch]
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }


# ── 학습된 Self-RAG (ModelRAG 상속) ─────────────────────────────────────────

def _extract_answer_from_selfrag_output(decoded: str) -> str:
    """
    Self-RAG 생성 결과에서 답변만 추출 (평가용)
    [Generate] 와 [Fully supported]/[Partially supported]/[No support] 사이의 텍스트
    """
    gen_marker = "[Generate]"
    end_markers = ["[Fully supported]", "[Partially supported]", "[No support]"]
    if gen_marker in decoded:
        start = decoded.find(gen_marker) + len(gen_marker)
        rest = decoded[start:].strip()
        end_pos = len(rest)
        for m in end_markers:
            if m in rest:
                end_pos = min(end_pos, rest.find(m))
        return rest[:end_pos].strip()
    return decoded.strip()


class ModelRAGTrainedSelfRAG(ModelRAG):
    """
    논문 정렬 Self-RAG Generator
    - 적응형 검색: [Retrieval] vs [No Retrieval] 모델이 선택
    - 생성 중 reflection: [Relevant], [Fully supported], [Utility:X] 등 출력
    - 반복적 검색: 생성 중 [Retrieval] 재출력 시 추가 검색
    """

    def _build_context(self, passages: List[dict]) -> str:
        return _build_context(passages)

    def _format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """논문 형식: ### Instruction + ### Response"""
        prompt = INSTRUCTION_TEMPLATE.format(question=question)
        if context:
            prompt += CONTEXT_TEMPLATE.format(context=context)
        return prompt

    def decode_outputs_for_eval(self, outputs: torch.Tensor) -> List[str]:
        """평가 시 Self-RAG 출력에서 답변만 추출 (eval_for_rag에서 사용)"""
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return [_extract_answer_from_selfrag_output(d) for d in decoded]

    def _generate_single_iterative(
        self,
        query: str,
        qid: str,
        k: int,
        max_new_tokens: int,
        max_retrieval_rounds: int,
        pad_token_id: int,
        gen_kw: dict,
    ) -> List[int]:
        """단일 쿼리에 대해 적응형·반복적 Self-RAG 생성"""
        prompt = INSTRUCTION_TEMPLATE.format(question=query)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = input_ids.to(self.model.device)

        total_generated = 0
        retrieval_rounds = 0
        generated_ids = []

        while total_generated < max_new_tokens:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                pad_token_id=pad_token_id,
                do_sample=gen_kw.get("do_sample", False),
                temperature=gen_kw.get("temperature", 1.0),
            )
            next_token_id = outputs[0, -1].item()
            next_token = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
            generated_ids.append(next_token_id)
            input_ids = outputs
            total_generated += 1

            if next_token_id == self.tokenizer.eos_token_id:
                break

            if "[Retrieval]" in next_token or next_token.strip() == "[Retrieval]":
                if retrieval_rounds < max_retrieval_rounds:
                    retrieval_rounds += 1
                    search_query = query
                    if generated_ids:
                        partial = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        if partial.strip():
                            search_query = f"{query} {partial}"
                    list_passages, _ = self.search([search_query], [qid], k=k)
                    if list_passages and list_passages[0]:
                        passages = list_passages[0]
                        context = self._build_context(passages)
                        ctx_ids = self.tokenizer.encode(
                            "\n" + context,
                            add_special_tokens=False,
                            return_tensors="pt",
                        ).to(self.model.device)
                        max_ctx = 992 - input_ids.size(1)
                        if max_ctx > 0 and ctx_ids.size(1) > max_ctx:
                            ctx_ids = ctx_ids[:, :max_ctx]
                        if ctx_ids.size(1) > 0:
                            input_ids = torch.cat([input_ids, ctx_ids], dim=1)

            elif "[No Retrieval]" in next_token or next_token.strip() == "[No Retrieval]":
                pass

        return generated_ids

    @torch.no_grad()
    def retrieval_augmented_generate(
        self,
        queries: List[str],
        qids: List[str],
        k: int = 5,
        max_new_tokens: int = 64,
        max_retrieval_rounds: int = 3,
        **kwargs,
    ) -> torch.Tensor:
        """
        논문 Self-RAG 인퍼런스:
        - 모델이 [Retrieval] / [No Retrieval] 선택 (적응형 검색)
        - 생성 중 [Relevant] 등 reflection 토큰 출력
        - [Retrieval] 재출력 시 추가 검색 (반복적 검색)
        """
        try:
            import transformers
            pad_token_id = (
                getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.model.config, "eos_token_id", None)
                or 0
            )
        except ImportError:
            pad_token_id = 0

        gen_kw = {k: v for k, v in kwargs.items() if k in ("do_sample", "temperature")}

        list_generated = []
        for query, qid in zip(queries, qids):
            ids = self._generate_single_iterative(
                query, qid, k, max_new_tokens, max_retrieval_rounds, pad_token_id, gen_kw
            )
            list_generated.append(ids)

        max_len = max(len(g) for g in list_generated)
        pad_id = pad_token_id if pad_token_id is not None else self.tokenizer.eos_token_id
        padded = [
            g + [pad_id] * (max_len - len(g))
            for g in list_generated
        ]
        return torch.tensor(padded, dtype=torch.long, device=self.model.device)


# ── 학습 함수 ─────────────────────────────────────────────────────────────

def train_selfrag(
    model,
    tokenizer,
    train_dataset: SelfRAGDataset,
    retriever,
    train_config: dict,
    eval_dataset=None,
    eval_loop=None,
):
    """Self-RAG Generator 학습"""
    from torch.utils.data import DataLoader

    add_selfrag_tokens(tokenizer, model)
    collator = SelfRAGCollator(tokenizer, max_length=train_config.get("max_length", 992))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 8),
        shuffle=True,
        collate_fn=collator,
    )

    model = model.to(train_config.get("device", "cuda"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 5e-5),
        weight_decay=train_config.get("weight_decay", 0.01),
    )

    model.train()
    for epoch in range(train_config.get("num_epochs", 3)):
        for batch in train_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
