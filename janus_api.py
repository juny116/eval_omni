# server_with_guided_choice.py
# Drop-in replacement for your server.py with vLLM-like `guided_choice` support
# (OpenAI-compatible extra param). Key additions are marked with "# [GUIDED]".

import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Set

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor

# ── 모델 선택 ──────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("JANUS_MODEL_ID", "deepseek-ai/Janus-Pro-7B")

# Janus의 대화 템플릿/토크나이저
from janus.models import MultiModalityCausalLM, VLChatProcessor

# ── Janus 준비 (Tokenizer/Processor) ───────────────────────────────────────
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_ID)
tokenizer = vl_chat_processor.tokenizer  # transformers.PreTrainedTokenizerFast

# ── Accelerate 기반: 멀티-GPU 자동 배치 ─────────────────────────────────────
max_mem_gb = os.environ.get("JANUS_GPU_MEM_GB")
max_memory = None
if max_mem_gb is not None:
    try:
        gb = int(max_mem_gb)
        if torch.cuda.is_available():
            max_memory = {i: f"{gb}GiB" for i in range(torch.cuda.device_count())}
    except ValueError:
        pass

# trust_remote_code=True 는 Janus의 커스텀 모듈 로딩에 필요
model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
)
model.eval()

# ── FastAPI 앱 ───────────────────────────────────────────────────────────
app = FastAPI(title="Janus-Pro OpenAI Compatible Server")

# ── OpenAI 호환 스키마 ─────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False  # 이 예시는 스트리밍 미지원
    # [GUIDED] — vLLM과 동일한 파라미터 이름
    guided_choice: Optional[List[str]] = None

class Choice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]

# --- (스키마: OpenAI Completions) ---
class CompletionRequest(BaseModel):
    model: Optional[str] = None
    # ⬇️ 문자열, 문자열 리스트, 토큰ID 리스트, 토큰ID 2중리스트 모두 허용
    prompt: Optional[Union[str, List[str], List[int], List[List[int]]]] = ""
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # [GUIDED]
    guided_choice: Optional[List[str]] = None

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: str

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

# ── 유틸: 프롬프트 빌드(텍스트 전용) ───────────────────────────────────────
def build_prompt(messages: List[ChatMessage]) -> str:
    msgs = list(messages)  # 사이드이펙트 방지
    if msgs and msgs[-1].role == "user":
        msgs.append(ChatMessage(role="assistant", content=""))

    conv = []
    sys_buf = []
    for m in msgs:
        if m.role == "system":
            sys_buf.append(m.content)
        elif m.role == "user":
            conv.append({"role": "<|User|>", "content": m.content})
        elif m.role == "assistant":
            conv.append({"role": "<|Assistant|>", "content": m.content})

    system_prompt = "\n".join(sys_buf) if sys_buf else ""
    sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=system_prompt,
    )
    return sft

# ── 유틸: 토큰 기반 stop 트렁케이션 ────────────────────────────────────────
def _encode_stop_variants(stop: str) -> List[List[int]]:
    variants = [stop, " " + stop, "\n" + stop]
    seqs = []
    for s in variants:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    uniq = []
    seen = set()
    for s in seqs:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq

def _find_stop_pos_ids(gen_ids: List[int], stop_seqs: List[List[int]]) -> Optional[int]:
    if not gen_ids or not stop_seqs:
        return None
    n = len(gen_ids)
    for k_seq in stop_seqs:
        m = len(k_seq)
        if m == 0 or m > n:
            continue
        for i in range(0, n - m + 1):
            if gen_ids[i:i+m] == k_seq:
                return i
    return None

def _byte_offsets_from_tokens(token_ids: List[int]) -> List[int]:
    offsets = []
    offset = 0
    for tid in token_ids:
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        offsets.append(offset)
        offset += len(s.encode("utf-8"))
    return offsets

# [GUIDED] ────────────────────────────────────────────────────────────────
# Token-ID Trie + LogitsProcessor to enforce that the generated text equals one of given choices
class _TrieNode:
    __slots__ = ("children", "terminal")
    def __init__(self) -> None:
        self.children: Dict[int, "_TrieNode"] = {}
        self.terminal: bool = False

class TokenTrie:
    def __init__(self, sequences: List[List[int]]):
        self.root = _TrieNode()
        for seq in sequences:
            self._insert(seq)

    def _insert(self, seq: List[int]):
        node = self.root
        for tid in seq:
            node = node.children.setdefault(tid, _TrieNode())
        node.terminal = True

    def allowed_next(self, prefix: List[int]) -> Tuple[Set[int], bool]:
        """
        Returns (allowed_token_ids, is_terminal_prefix)
        - allowed_token_ids: if empty and is_terminal_prefix=True, caller should force EOS
        - is_terminal_prefix: True if prefix already matches a complete choice
        """
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return set(), False
            node = node.children[tid]
        return set(node.children.keys()), node.terminal

class GuidedChoiceLogitsProcessor(LogitsProcessor):
    def __init__(self, trie: TokenTrie, eos_token_id: int, prefix_len: int):
        self.trie = trie
        self.eos = eos_token_id
        self.prefix_len = prefix_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # batch size 1 가정 (이 데모 서버는 n=1)
        ids = input_ids[0].tolist()
        gen_prefix = ids[self.prefix_len:]
        allowed, done = self.trie.allowed_next(gen_prefix)
        # 이미 완성된 선택지와 정확히 일치 → EOS 강제
        if done:
            scores[:] = float('-inf')
            scores[..., self.eos] = 0.0
            return scores
        # 불일치(선택지 트라이에서 벗어남) → 생성 불가. (전체 -inf 방지 위해 EOS 허용)
        if not allowed:
            scores[:] = float('-inf')
            scores[..., self.eos] = 0.0
            return scores
        mask = torch.full_like(scores, float('-inf'))
        mask[..., list(allowed)] = 0.0
        return scores + mask

# 유틸: guided_choice 문자열 → 토큰 시퀀스(공백/개행 변형 포함)
def _encode_choice_variants(choice: str) -> List[List[int]]:
    variants = [choice, " " + choice, "\n" + choice]
    seqs: List[List[int]] = []
    for s in variants:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    # 중복 제거
    uniq: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = set()
    for s in seqs:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            uniq.append(s)
    return uniq

# ── /v1/chat/completions ───────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request) -> Any:
    if req.stream:
        raise ValueError("stream=true is not supported in this demo server.")

    prompt = build_prompt(req.messages)
    inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inp.input_ids  # (1, T_in)

    lm = model.language_model
    lm_device = next(lm.parameters()).device

    # OpenAI SDK extra_body 지원: extra_body.guided_choice → guided_choice
    _guided_choice = getattr(req, "guided_choice", None)
    try:
        raw = await request.json()
        if isinstance(raw, dict):
            _guided_choice = raw.get("guided_choice") or (raw.get("extra_body") or {}).get("guided_choice") or _guided_choice
    except Exception:
        pass

    # [GUIDED] logits processor 구성 (있을 때만)
    logits_procs: Optional[List[LogitsProcessor]] = None
    if _guided_choice:
        # 선택지 트라이 구성
        all_variants: List[List[int]] = []
        for c in _guided_choice:
            all_variants.extend(_encode_choice_variants(c))
        trie = TokenTrie(all_variants)
        logits_procs = [GuidedChoiceLogitsProcessor(trie, tokenizer.eos_token_id, prefix_len=input_ids.shape[-1])]

    with torch.inference_mode():
        out = lm.generate(
            input_ids=input_ids.to(lm_device),
            max_new_tokens=req.max_tokens or 512,
            do_sample=(req.temperature or 0.0) > 0,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 1.0,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=logits_procs,
        )

    seq = out.sequences[0].tolist()
    gen_ids = seq[input_ids.shape[-1]:]

    stop_list = req.stop or []
    stop_candidates = []
    for s in stop_list:
        stop_candidates.extend(_encode_stop_variants(s))
    cut_idx = _find_stop_pos_ids(gen_ids, stop_candidates)
    if cut_idx is not None:
        gen_ids = gen_ids[:cut_idx]

    generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    prompt_tokens = int(input_ids.numel())
    completion_tokens = len(tokenizer.encode(generated))
    total_tokens = prompt_tokens + completion_tokens

    resp = ChatResponse(
        id="chatcmpl-januspro-accel",
        object="chat.completion",
        created=int(time.time()),
        model=req.model or MODEL_ID,
        choices=[
            Choice(
                index=0,
                message={"role": "assistant", "content": generated},
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    )
    return resp

# ── (원본 로짓/생성 유틸 그대로) ───────────────────────────────────────────

def _prompt_logprobs(input_ids: torch.Tensor, lm, top_k: Optional[int]):
    B, T = input_ids.shape
    first_dev = next(lm.parameters()).device
    input_ids = input_ids.to(first_dev)
    with torch.inference_mode():
        out = lm(input_ids=input_ids, use_cache=False)
        logits = out.logits  # (B, T, V)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        next_ids = input_ids[:, 1:]  # (B, T-1)
        gather_logp = logp.gather(-1, next_ids.to(logp.device).unsqueeze(-1)).squeeze(-1)

    results = []
    for b in range(B):
        ids = input_ids[b].tolist()
        token_logprobs = [None] + [float(x) for x in gather_logp[b].tolist()]
        if top_k and top_k > 0:
            top_vals, top_idx = torch.topk(logp[b], k=min(top_k, logp.shape[-1]), dim=-1)
            top_list = []
            for t in range(top_vals.shape[0]):
                d = {tokenizer.convert_ids_to_tokens(int(i)): float(v) for i, v in zip(top_idx[t], top_vals[t])}
                top_list.append(d)
            top_logprobs = [None] + top_list
        else:
            top_logprobs = None
        text_offset = _byte_offsets_from_tokens(ids)
        results.append({
            "tokens": [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in ids],
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": text_offset,
        })
    return results


def _gen_with_logprobs(lm, input_ids: torch.Tensor, gen_kwargs, top_k: Optional[int]):
    first_dev = next(lm.parameters()).device
    input_ids = input_ids.to(first_dev)
    with torch.inference_mode():
        out = lm.generate(input_ids=input_ids, return_dict_in_generate=True, output_scores=True, **gen_kwargs)
    seq = out.sequences  # (B, T_total)
    new_len = len(out.scores)
    if new_len == 0:
        return seq, {"ids": [], "tokens": [], "token_logprobs": [], "top_logprobs": [] if (top_k and top_k > 0) else None, "text_offset": []}
    scores = torch.stack(out.scores, dim=1)      # (B, T_new, V)
    logp = F.log_softmax(scores, dim=-1)         # (B, T_new, V)
    gen_token_ids = seq[:, -new_len:]
    gather_logp = logp.gather(-1, gen_token_ids.to(logp.device).unsqueeze(-1)).squeeze(-1)
    top_list_all = None
    if top_k and top_k > 0:
        top_vals, top_idx = torch.topk(logp, k=min(top_k, logp.shape[-1]), dim=-1)
        top_list_all = []
        for t in range(new_len):
            step = []
            for b in range(top_vals.shape[0]):
                d = {tokenizer.convert_ids_to_tokens(int(idx)): float(val) for idx, val in zip(top_idx[b, t], top_vals[b, t])}
                step.append(d)
            top_list_all.append(step)
    ids = [int(x) for x in gen_token_ids[0].tolist()]
    tokens = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]
    tlogp = [float(x) for x in gather_logp[0].tolist()]
    if top_list_all is not None:
        top_logprobs = [top_list_all[t][0] for t in range(len(tokens))]
    else:
        top_logprobs = None
    text_offset = _byte_offsets_from_tokens(ids)
    return seq, {"ids": ids, "tokens": tokens, "token_logprobs": tlogp, "top_logprobs": top_logprobs, "text_offset": text_offset}

# --- (핵심) /v1/completions 엔드포인트 ---
@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request) -> Any:
    # OpenAI SDK extra_body 지원: extra_body.guided_choice → guided_choice
    _guided_choice = getattr(req, "guided_choice", None)
    try:
        raw = await request.json()
        if isinstance(raw, dict):
            _guided_choice = raw.get("guided_choice") or (raw.get("extra_body") or {}).get("guided_choice") or _guided_choice
    except Exception:
        pass
    if req.stream:
        raise ValueError("stream=true is not supported in this demo server.")

    text_prompts: List[str] = []
    id_prompts: List[List[int]] = []

    p = req.prompt
    if isinstance(p, str):
        text_prompts = [p]
    elif isinstance(p, list):
        if not p:
            text_prompts = [""]
        else:
            if all(isinstance(x, str) for x in p):
                text_prompts = p
            elif all(isinstance(x, int) for x in p):
                id_prompts = [p]
            elif all(isinstance(x, list) for x in p):
                if all(all(isinstance(y, int) for y in x) for x in p):
                    id_prompts = p
                else:
                    raise ValueError("prompt list contains unsupported nested types.")
            else:
                raise ValueError("prompt list contains mixed/unsupported types.")
    elif p is None:
        text_prompts = [""]
    else:
        raise ValueError("Unsupported prompt type.")

    if req.n and req.n != 1:
        raise ValueError("This demo supports n=1 only.")
    if req.best_of and req.best_of != 1:
        raise ValueError("This demo supports best_of=1 only.")

    choices: List[CompletionChoice] = []
    total_prompt_toks = 0
    total_completion_toks = 0

    lm = model.language_model
    lm_device = next(lm.parameters()).device

    def _iter_prompts():
        for s in text_prompts:
            # vLLM의 텍스트 completion도 내부적으로 대화 템플릿이 더 안정적이라 동일 경로 사용
            msgs = [ChatMessage(role="user", content=s)]
            sft = build_prompt(msgs)
            inp = tokenizer(sft, return_tensors="pt", add_special_tokens=True)
            yield sft, inp.input_ids
        for ids in id_prompts:
            t = torch.tensor([ids], dtype=torch.long)
            s = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            yield s, t

    for idx, (ptxt, input_ids) in enumerate(_iter_prompts()):
        total_prompt_toks += int(input_ids.numel())
        lp_k = req.logprobs if (req.logprobs and req.logprobs > 0) else None
        echo_blob = None
        if req.echo or (req.max_tokens or 0) == 0:
            echo_blob = _prompt_logprobs(input_ids, lm, lp_k)[0]

        gen_text = ""
        gen_blob = {"ids": [], "tokens": [], "token_logprobs": [], "top_logprobs": None, "text_offset": []}
        finish_reason = "stop"

        stop_list: List[str] = []
        if isinstance(req.stop, str) and req.stop:
            stop_list = [req.stop]
        elif isinstance(req.stop, list):
            stop_list = [s for s in req.stop if s]
        stop_candidates = []
        for s in stop_list:
            stop_candidates.extend(_encode_stop_variants(s))

        # [GUIDED] — 필요 시 logits_processor 구성
        logits_procs: Optional[List[LogitsProcessor]] = None
        if _guided_choice:
            all_variants: List[List[int]] = []
            for c in _guided_choice:
                all_variants.extend(_encode_choice_variants(c))
            trie = TokenTrie(all_variants)
            logits_procs = [GuidedChoiceLogitsProcessor(trie, tokenizer.eos_token_id, prefix_len=input_ids.shape[-1])]

        if (req.max_tokens or 0) > 0:
            gen_kwargs = dict(
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature or 0.0) > 0,
                temperature=req.temperature or 1.0,
                top_p=req.top_p or 1.0,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                logits_processor=logits_procs,
            )
            seq, gen_blob = _gen_with_logprobs(lm, input_ids, gen_kwargs, lp_k)
            full_ids = seq[0].tolist()
            gen_ids = gen_blob["ids"]
            cut_idx = _find_stop_pos_ids(gen_ids, stop_candidates)
            if cut_idx is not None:
                gen_ids = gen_ids[:cut_idx]
                gen_blob["ids"] = gen_ids
                gen_blob["tokens"] = gen_blob["tokens"][:cut_idx]
                gen_blob["token_logprobs"] = gen_blob["token_logprobs"][:cut_idx]
                if gen_blob["top_logprobs"] is not None:
                    gen_blob["top_logprobs"] = gen_blob["top_logprobs"][:cut_idx]
                gen_blob["text_offset"] = gen_blob["text_offset"][:cut_idx]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            total_completion_toks += len(tokenizer.encode(gen_text))

        if req.echo:
            text = ptxt + gen_text
            if req.logprobs and req.logprobs > 0:
                prompt_top = echo_blob["top_logprobs"]
                if prompt_top is None:
                    prompt_top = [None] * len(echo_blob["tokens"])
                gen_top = gen_blob["top_logprobs"]
                if gen_top is None:
                    gen_top = [None] * len(gen_blob["tokens"])
                logprobs_obj = {
                    "tokens": (echo_blob["tokens"] + gen_blob["tokens"]),
                    "token_logprobs": (echo_blob["token_logprobs"] + gen_blob["token_logprobs"]),
                    "top_logprobs": (prompt_top + gen_top),
                    "text_offset": (echo_blob["text_offset"] + gen_blob["text_offset"]),
                }
            else:
                logprobs_obj = None
        else:
            text = gen_text
            if req.logprobs and req.logprobs > 0:
                gen_top = gen_blob["top_logprobs"]
                if gen_top is None:
                    gen_top = [None] * len(gen_blob["tokens"])
                logprobs_obj = {
                    "tokens": gen_blob["tokens"],
                    "token_logprobs": gen_blob["token_logprobs"],
                    "top_logprobs": gen_top,
                    "text_offset": gen_blob["text_offset"],
                }
            else:
                logprobs_obj = None

        choices.append(CompletionChoice(text=text, index=idx, logprobs=logprobs_obj, finish_reason=finish_reason))

    resp = CompletionResponse(
        id="cmpl-januspro-local",
        object="text_completion",
        created=int(time.time()),
        model=req.model or MODEL_ID,
        choices=choices,
        usage={
            "prompt_tokens": total_prompt_toks,
            "completion_tokens": total_completion_toks,
            "total_tokens": total_prompt_toks + total_completion_toks,
        },
    )
    return resp
