import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import transformers
from typing import Callable, List, Optional, Tuple, Union

class TransformerConfig(transformers.PretrainedConfig):
    model_type = "custom_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: Optional[int] = None,
        max_postion_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_postion_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout

        if self.head_dim is None:
            self.head_dim = hidden_size // num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        assert self.hidden_size == (self.num_attention_heads * self.head_dim), "hidden_size must be equal to num_attention_heads * head_dim"

def apply_rotary_emb(x, position_embeddings):
    cos,sin = position_embeddings
    # fill here: rotate hidden states with positon angle
    # shape hint
    # cos,sin: (batch_size, seq_len, head_dim // 2)
    # x: (batch_size, num_attention_heads, seq_len, head_dim)
    # x.pow(2) : 거듭제곱 함수 -> x 의 모든 원소를 제곱하라는 뜻
    #RMS Norm : 각 토큰의 히든 스테이트 벡터가 너무 크거나 작아지지 않도록 크기를 유지시켜줌, Transformer 각 Layer 안에서 두번 사용
    #keepdim=True : 차원 축소 방지
    #(batch, seq, hidden dim) 은 embedding 거치면 행렬상 이렇게 바뀜 ID -> 벡터 되면서
    #######
    cos, sin = position_embeddings #튜플로 저장

    # head dim 을 절반으로 나눠
    x1 = x[..., :x.shape[-1] //2 ] #...은 앞에 있는 차원은 x 에서 전부 가져와, 마지막 차원만 자르겠다
    x2 = x[..., x.shape[-1] // 2:] #뒤 절반

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1) #broadcasting 으로 차원을 (batch, 1, seq_len, head_dim//2) 로 맞춤

    rotated = torch.cat([-x2, x1], dim=-1)
    x = x*cos + rotated*sin
    #######
    return x

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # fill here: rms normalization
        #######
        input_dtype = x.dtype #텐서의 데이터타입으로 변환
        x = x.float() #32로 올려서 안전하게 계산
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = x / rms
        output = output.to(input_dtype)
        output = output.to(input_dtype) * self.weight
        
        #######
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class RotaryEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.config = config
        
        # fill here: rotary embedding initalization
        # shape hint
        # inv_freq: (1, head_dim // 2)
        # unsqueeze(0) : 0번 위치에 차원 하나 추가 (head_dim//2,) -> (1, head_dim//2)
        ######
        head_dim = config.head_dim
        theta = config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim))
        ######

        self.register_buffer("inv_freq", inv_freq, persistent=False) #모델 안에 텐서 저장하는 방법은 nn.Parameter (학습되는 값), register_buffer (학습되지 않는 값)
    #model.state_dict() - 키 벨류로 이뤄진 딕셔너리로 저장
    @torch.no_grad() #이 함수 내에서는 gradient 계산하지 않겠다는 것 학습파라미터가 아닌 위치에 따라 고정된 값이니까
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) #처음에 2->4차원 확장,float() 32차원
        '''
        .expand() 텐서 복제 하지 말고 반복 확장. 메모리 추가로 안쓰고 shape 만 늘려
        x = torch.tensor([1,2,3]) 이라면
        x.expand(4,-1) 은 [1,2,3] 을 차원 유지하면서 4번 반복
        .expand(position_ids.shape[0], -1, 1) 은 배치 크기만큼 반복
        이때 -1 부분인 두번째 부분은 그대로 유지, 1 인 세번째 부분은 1로 고정
        ex. 현재 (1,16,1) 이면 .expand(3,-1,1) 은 (3,16,1) 이 됨
        position_ids 처음 차원은 (배치 사이즈, 문장 길이) -> 거기다가 None 추가로 1 추가

        freqs 는 회전 속도 * 위치로 각 위치, 차원쌍의 회전 각도 (라디안)
        inv_freq 는 위치가 1 증가시 얼마나 회전하나 옆이면 빠른 속도, 멀면 느린 속도

        head_dim = 8일 때:

        arange(0, 8, 2) = [0, 2, 4, 6]
        / head_dim       = [0/8, 2/8, 4/8, 6/8] = [0, 0.25, 0.5, 0.75]
        10000 **         = [1, 10, 100, 1000]
        1.0 /            = [1.0, 0.1, 0.01, 0.001]   ← 점점 느려짐
        '''
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False): # disable autocasting for fp32 precision
            # fill here: calculate rotary embedding
            # shape hint
            # cos,sin: (batch_size, seq_len, head_dim // 2)
            #
            ######
            #(batch, head_dim//2, 1) @ (batch, 1, seq_len) -> (batch, head_dim//2, seq_len)
            freqs = inv_freq_expanded @ position_ids_expanded #절대 각도. 이후 Q*K 내적으로 모든 상대 각도 계산
            # (batch, head_dim//2, seq_len) -> (batch, seq_len, head_dim//2)
            freqs = freqs.transpose(1,2)
            cos = freqs.cos()
            sin = freqs.sin()
            ######

        return cos.to(x.dtype), sin.to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.head_dim = config.hidden_size // config.num_attention_heads #각 헤드가 담당하는 attention 차원 크기
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads #메모리 절약, 여러 헤드가 하나의 키 벨류
        self.scale = self.head_dim**-0.5 
        self.dropout = config.attention_dropout #비율에 맞춰 0으로 바꿈 

        #fill here: initalization of query, key, value and output projection
        #q,k,v 각 projection 으로 쿼리 키 벨류 역할 부여
        ######
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        ######

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], #ROPE 에서 계산한 값, Tuple 은 두 값이 묶여 있다는 것것
        attention_mask: Optional[torch.Tensor],#값이 있을수도, 없을 수도 있음음
        past_key_value: Optional[List[Tuple[torch.Tensor,torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        # batch , num_head, seq_len, head_dim
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        

        query_states = apply_rotary_emb(query_states, position_embeddings)
        key_states = apply_rotary_emb(key_states, position_embeddings)

        if past_key_value is not None:
            key_cache, value_cache = past_key_value
            key_states = torch.cat([key_cache, key_states], dim=-2)
            value_states = torch.cat([value_cache, value_states], dim=-2)
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # fill here: calculate attention weights
        # shape hint
        # qkv states: (batch_size, num_attention_heads, seq_len, head_dim)
        # query: (batch, num_heads, seq_len, head_dim)
#key^T: (batch, num_heads, head_dim, seq_len)   ← transpose(2,3) 후

#matmul 결과:
#(batch, num_heads, seq_len, seq_len)  ← attn_weights의 실제 shape
#                    ↑        ↑
#                  나(토큰)  상대(토큰)
        
    # matmul : 행렬 곱셈 연산, torch.matmul 은 마지막 두 차원만 행렬곱하고 앞차원은 그냥 묶음 취급 (batch × num_heads) 개 행렬 곱 동시수행
    
    
        #######
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask #더하면서 마스크 적용 부분에 -inf 가 들어 있던 거를 더하게 되어 자동으로 무시하게 만듦

        #softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        #Dropout 
        attn_weights = F.dropout(attn_weights, p=self.dropout, training = self.training)

        #Attention * Value
        attn_output = torch.matmul(attn_weights, value_states) #맥락간 점수 가중치로 문맥 파악시킴

        # (batch, num_heads, seq_len, head_dim) -> reshape : batch, seq_len, hidden_size 
        # dim=-1은 마지막 차원 방향으로 연산 마지막 차원 방향으로 확률합 1이 되게게

        attn_output = attn_output.transpose(1,2).contiguous() #메모리 읽는걸 transpose 한 방향으로 바꿔서 메모리 효율화화
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        #######
        return attn_output, past_key_value

# FFN : 실제 의미를 변환 및 처리 
# 처리시 일부러 차원 늘렸다 줄임 -> 복잡한 정보 처리 
# gate - 입력에 따라 동적으로 중요 차원만 선택
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False) #nn.Linear 은 히든층 없다 하지만 FFN 하면 중간 up 한 부분이 히든층
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,bias=False)
        self.act_fn = torch.nn.SiLU()
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, x):
        # fill here: feed forward network
        ######
        gate = self.act_fn(self.gate_proj(x)) #silu 붙어서 스위치/밸브 역할로 학습
        up = self.up_proj(x) #순수 선형 변환 upscaling
        outputs = self.down_proj(gate*up) #gate*up 으로 필터링함 정보 -> 이걸 원래 차원으로 압축
        outputs = self.dropout(outputs)
        
        ######
        return outputs

class TransformerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config)

        self.feed_forward = FeedForwardNetwork(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # fill here: transformer layer
        # return layer hidden states and past key values with output of attention
        ######
        
        residual = hidden_states #원본 백업
        hidden_states = self.input_layernorm(hidden_states) #정규화

        hidden_states, past_key_value = self.self_attn(
            hidden_states = hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = attention_mask,
            past_key_value = past_key_value,
        ) #정규화된 값으로 self attention, 변수 이름 동일하게 한거

        #forward 마지막줄 보면 return attn_output, past_key_value 이므로 반환하라고 할당
        #hidden_states 가 attention 거친 변환된 표현
        #past_key_value 는 attention 에서 사용된 키와 값의 캐시

        hidden_states = residual + hidden_states #원본 + self attention 결과 (잔차연결 add) 연속해서 파이프라인 처리

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        ######
        return hidden_states, past_key_value

class TransformerPreTrainedModel(transformers.PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
    ):
        batch_size, seq_len = input_ids.shape
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        if inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or input_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=inputs_embeds.device).unsqueeze(0)

        target_length = seq_len
        seen_token_length = 0
        if past_key_values is not None:
            seen_token_length = past_key_values[0][0].shape[-2]
            target_length += seen_token_length
        
        attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=target_length,
            seen_token_length=seen_token_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            batch_size=batch_size,
        )

        hidden_states = inputs_embeds
        position_embed = self.rotary_emb(hidden_states, position_ids)
        kv_cache_new = []
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values[layer_idx] if past_key_values is not None else None,
                    position_embed
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                    position_embeddings=position_embed,
                )

            hidden_states, kv_cache = layer_outputs
            kv_cache_new.append(kv_cache)

        hidden_states = self.norm(hidden_states)

        if past_key_values is not None:
            past_key_values = kv_cache_new

        return hidden_states, past_key_values

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        seen_token_length: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int
    ):
        # fill here: prepare attention mask
        # shape hint
        # mask: (batch_size, 1, sequence_length, target_length)
        ######
        
        #casual mask: 미래 토큰 못보게 차단
        casual_mask = torch.full(
            (sequence_length, target_length),
            fill_value=torch.finfo(dtype).min,#설정된 dtype 에서 가장 작은 수로 반환환
            dtype=dtype,
            device=device,
        )
        #seq_len : 지금 입력하는 토큰 수 새 토큰
        #target length: 는 볼 수 있는 전체 토큰 수 즉 과거
        # 학습시는 한번에 다, 생성시는 이전 생성한 것, 추후 생성하는 것 따로
        #볼 수 있는 위치 (자기 자신 + 이전 토큰)는 0으로
        #seen_token_length: KV cache 가 있을 때 이미 처리된 길이
        #torch.triu 는 대각선 위 값 빼고 전부 0
        if sequence_length !=1:
            casual_mask = torch.triu(casual_mask, diagonal=1) #diagonal=1 은 대각선 위의 값을 0으로 만듦
            #마스크 씌운게 -inf 해서 못보게 만들기
        
        if seen_token_length > 0:
            casual_mask[:, :seen_token_length] = 0 #이미 처리된 부분은 0으로 만듦 그래야 이전 과정 볼 수 있잖아
        
        #패딩 마스크 반영
        mask = casual_mask.unsqueeze(0).unsqueeze(0) #앞에다가 크기 1인 차원 추가
        mask = mask.expand(batch_size, 1, sequence_length, target_length)
        mask = mask.clone()

        if attention_mask is not None:
            #패딩 위치 0 -> -inf, 실제 위치 1 을 0으로 변환
            pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(dtype).min
            #attention_mask[:, None, None, :].float()) 은 unsqueeze 와 같아서 1 추가 곧 차원확장 -> (batch, 1, 1, target_length)
            #target_length 기준으로 패딩 마스크 

            #pad mask shape 가 (batch, 1, 1, target_length) 인 이유는 Attention 가중치 shape 가 (batch, num_heads, seq_len, seq_len) 이기 때문
            # 여기에다 더해야 하므로 shape 호환 후 broadcasting 으로 확장
            pad_mask_full = torch.zeros(batch_size, 1, sequence_length, target_length, dtype=dtype, device=device)
            pad_mask_full[:, :, :, seen_token_length:] += (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(dtype).min
            mask = mask + pad_mask_full
            mask = torch.clamp(mask, min=torch.finfo(dtype).min) #clamp 로 특정 범위 안에 값 유지 제한

        return mask  # (batch_size, 1, sequence_length, target_length)

#Optional[X]는 "X 타입이거나 None일 수 있다" 는 뜻입니다. 
# #즉 Optional[torch.Tensor] = torch.Tensor | None.
#왜 필요한가? 상황에 따라 안 넘기는 파라미터가 있기 때문
#예를 들어 use_cache 는 생성 시에만 사용되므로 학습시에는 넘기지 않음
#이런 경우 Optional[bool] = None 로 선언하면 학습시에는 None 이 되어 무시됨


class TransformerForCausalLM(TransformerPreTrainedModel): #다음 토큰 예측 모델
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ):  
        if use_cache and past_key_values is None:
            batch_size, _ = input_ids.shape
            #dummy 는 빈 텐서 형성 형태만 연산가능하게, 값은 무의미 임시값
            dummy = torch.empty((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).to(self.model.layers[0].self_attn.q_proj.weight)
            past_key_values = [(dummy.clone(),dummy.clone()) for _ in range(self.config.num_hidden_layers)]
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            logits = logits.float() # cast to fp32 for calculating softmax in high precision

            # fill here: calculate cross entropy loss
            # loss must be scalar
            ## TransformerForCausalLM에서
            #hidden_states  # (batch, seq_len, hidden_size)
            #    ↓ lm_head = nn.Linear(hidden_size, vocab_size)
            #logits         # (batch, seq_len, vocab_size)

            
            ######
            '''
            inputs['labels'] = inputs['input_ids'].clone()
            # input_ids: (batch, seq_len)
            # labels:    (batch, seq_len)  ← 2차원
            ```

            ---

            ### logits vs labels 차원 비교
            ```
            logits: (batch, seq_len, vocab_size)  ← 3차원
                    각 토큰마다 50000개 확률값 필요하니까

            labels: (batch, seq_len)              ← 2차원
                    각 토큰마다 정답 ID 하나만 있으면 되니까
            ```

            ---

            ### 구체적으로 보면
            ```
            labels:
            [[  -100,  -100,  -100,  1234,  5678,  2  ],
            [  -100,  -100,  4321,  8765,     2,  0  ]]
            batch=2, seq_len=6

            숫자 하나 = 토큰 ID 하나
            -100 = 무시
            '''
            
            shift_logits = logits[..., :-1,:].contiguous() #logits 에서 마지막 토큰 제거 마지막 예측은 0 이니까
            shift_labels = labels[..., 1:].contiguous() #labels 에서 첫 토큰 제거 

            #Cross Entropy Loss 계산
            #ignore_index=-100: 패딩이나 프롬프트 위치는 손실 계산에서 제외
            #view -> shape 변환 shift_logits 가 (batch=2, seq_len=5, vocab=500) 이면 view(-1, vocab_size) = (10,500)
            '''
            predictions: (N, C)  ← 2D만 받음
            targets:     (N,)    ← 1D만 받음
            N: 예측 개수 C: 클래스 단어 수
            '''
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) #-100 인 인덱스는 손실계산 제외

            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            ) #스칼라 변환 위해서
            
            ######
        return (loss,logits) if loss is not None else (logits, past_key_values)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        max_new_tokens: int = 32,
        return_response_only: bool = False,
    ):
        batch_size, init_seq_len = input_ids.shape
        device = input_ids.device
        eos = self.config.eos_token_id

        unfinish_flag = torch.ones(batch_size, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, -1)
        
        for _ in range(max_new_tokens):
            logits, past_key_values = self.forward(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids[:, -1:] if past_key_values is not None else position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinish_flag + eos * (1 - unfinish_flag)
            unfinish_flag = unfinish_flag * next_tokens.ne(eos)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)], dim=-1)

            if unfinish_flag.sum() == 0:
                break
        if return_response_only:
            return input_ids[:, init_seq_len:]
        return input_ids

class TransformerForSequenceClassification(TransformerPreTrainedModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = TransformerModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        inputs_embeds: torch.FloatTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.classifier(hidden_states[:, -1, :])
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction="mean")
        return (loss, logits) if loss is not None else (logits, past_key_values)