"""
Improved RAG: Context Compression (Instruction-aware)

Handout Section 4.3 - Context compression [6]
- 질문에 맞게 검색된 passage를 압축하여 노이즈 제거, Lost-in-the-middle 완화
- Few-shot (BaselineRAG) 상속 + Semantic reranking + Sentence-level extraction

필요 패키지: pip install sentence-transformers
"""

from baseline_rag import BaselineRAG
import numpy as np
from typing import List, Optional
import re

#a : (dim,) -> 질문 벡터 1개
#b: (n, dim) -> 문서 벡터 n개
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (dim,), b: (n, dim) -> (n,)"""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9) #axis=1 이면 각 행을 하나의 벡터로 보고 노름 계산, keepdims 해야 차원 1로 유지
    return np.dot(b_norm, a_norm)


def _split_sentences(text: str) -> List[str]:
    """문장단위로 분리하는 함수"""
    if not text or not text.strip():
        return []
    # . ! ? 로 분리
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if len(s.strip()) > 15]


class ImprovedRAG(BaselineRAG):
    """
    Context Compression RAG (BaselineRAG 상속)
    - BM25로 더 많이 검색 (retrieve_k) → Semantic reranking → 상위 keep_k개
    - 각 passage 내 문장별 질문-관련도로 압축 
    - Few-shot 유지
    """

    def __init__(
        self,
        retrieve_k: int = 15,
        keep_k: int = 5,
        max_sentences_per_passage: int = 5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__()
        self.retrieve_k = retrieve_k
        self.keep_k = keep_k
        self.max_sentences_per_passage = max_sentences_per_passage
        self.embedding_model_name = embedding_model
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model_name)
            print(f"ImprovedRAG: sentence-transformers 로드 ({self.embedding_model_name})")
            return self._encoder
        except ImportError:
            raise ImportError("sentence-transformers 설치")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩 (batch)"""
        enc = self._get_encoder()
        return enc.encode(texts, convert_to_numpy=True)

    def _compress_passage(
        self,
        query: str,
        passage_text: str,
        top_k_sentences: int,
    ) -> str:
        """
        Instruction-aware: 질문에 관련된 문장만 추출
        """
        sentences = _split_sentences(passage_text)
        if len(sentences) <= top_k_sentences:
            return " ".join(sentences)

        query_emb = self._encode([query])[0]#그 실제 질문 벡터
        sent_embs = self._encode(sentences)
        scores = _cosine_similarity(query_emb, sent_embs)
        top_idx = np.argsort(scores)[-top_k_sentences:] #argsort 는 오름차순이니 큰거 뽑으려먼 -topk
        selected = [sentences[i] for i in sorted(top_idx)]
        return " ".join(selected)

    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        """
        Few-shot + Context compression
        1. BM25로 retrieve_k개 검색
        2. Semantic reranking (질문-문서 유사도) → 상위 keep_k개
        3. 각 passage 내 문장 압축 (질문 관련 문장만)
        4. Few-shot + 압축된 context로 프롬프트 구성
        """
        keep_k = min(k, self.keep_k)
        retrieve_k = max(self.retrieve_k, keep_k)

        list_passages, _ = self.search(queries, qids, k=retrieve_k)
        few_shot_text = self._build_few_shot_text()

        list_input_text = []
        for query, passages in zip(queries, list_passages):
            if not passages:
                input_text = f"{few_shot_text}\nQuestion: {query}\nAnswer:"
                list_input_text.append(input_text)
                continue

            # reranking
            passage_texts = [
                p.get("contents", p.get("text", ""))
                for p in passages
            ]
            query_emb = self._encode([query])[0]
            passage_embs = self._encode(passage_texts)
            scores = _cosine_similarity(query_emb, passage_embs)
            top_indices = np.argsort(scores)[-keep_k:][::-1] #뽑아서 거꾸로 정렬 가장 유사 passage 먼저
            #np.argsort 는 점수가 아니라 인덱스

            # compression per passage
            context_parts = [] #최종 context 에 들어갈 문자열 담은 리스트
            for idx in top_indices:
                p = passages[idx]
                title = p.get("title", "")
                text = p.get("contents", p.get("text", ""))
                compressed = self._compress_passage(
                    query,
                    text,
                    self.max_sentences_per_passage,
                )
                context_parts.append(f"Title: {title} Passage: {compressed}")

            context = "\n".join(context_parts)
            input_text = f"{few_shot_text}{context}\nQuestion: {query}\nAnswer:"
            list_input_text.append(input_text)

        return list_input_text
