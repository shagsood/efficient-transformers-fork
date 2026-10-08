"""QEfficient registration boundary for Hugging Face's HYV4 model.

The upstream HYV4 implementation owns the numerical definition. These classes
provide stable QEfficient identities while the MLA/DSA/iHC lowering is added in
focused follow-up changes. Keeping the classes as subclasses preserves the HF
forward contract for reduced-config parity tests.
"""

import torch
import torch.nn.functional as F

from transformers.models.hy_v4.modeling_hy_v4 import (
    HYV4Attention,
    HYV4DecoderLayer,
    HYV4Experts,
    HYV4ForCausalLM,
    HYV4HyperConnection,
    HYV4HyperHead,
    HYV4Indexer,
    HYV4MLP,
    HYV4Model,
    HYV4MoE,
    HYV4RMSNorm,
    HYV4RotaryEmbedding,
    HYV4TopkRouter,
    HYV4UnweightedRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.moe import _batched_linear
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from QEfficient.customop import ctx_gather_3d, ctx_scatter_3d
from QEfficient.transformers.cache_utils import QEffDynamicCompressedKVRopeLayer
from QEfficient.transformers.models._layerwise import (
    is_last_layer_window,
    is_layerwise_active,
    resolve_layer_window,
)


class QEffHYV4RMSNorm(HYV4RMSNorm):
    pass


class QEffHYV4UnweightedRMSNorm(HYV4UnweightedRMSNorm):
    pass


class QEffHYV4RotaryEmbedding(HYV4RotaryEmbedding):
    pass


class QEffHYV4Indexer(HYV4Indexer):
    pass


class QEffHYV4Cache:
    """Fixed-shape MLA + indexer cache used by the QEff ONNX ABI."""

    is_compileable = True

    def __init__(self, layers, indexer_keys, position_ids=None, layer_offset=0):
        self.layers = layers
        self.indexer_keys = indexer_keys
        self.position_ids = position_ids
        self.layer_offset = int(layer_offset)

    @classmethod
    def from_legacy_cache(cls, config, past_key_values, layer_offset=0):
        layers = []
        normalized = []
        for state in past_key_values:
            key, value = state[:2]
            # Keep the ABI tolerant of callers that construct legacy MLA pairs
            # in the HF (k_pe, compressed_kv) order.
            if key.shape[-1] == config.qk_rope_head_dim and value.shape[-1] == config.kv_lora_rank:
                key, value = value, key
            indexer = state[2] if len(state) > 2 else None
            normalized.append((key, value, indexer))
            layers.append(QEffDynamicCompressedKVRopeLayer(key, value))
        indexer_keys = [
            indexer
            if indexer is not None
            else torch.zeros((key.shape[0], key.shape[2], config.index_head_dim), dtype=key.dtype, device=key.device)
            for key, _, indexer in normalized
        ]
        return cls(layers, indexer_keys, layer_offset=layer_offset)

    def _local_layer(self, layer_idx):
        local_idx = int(layer_idx) - self.layer_offset
        if local_idx < 0 or local_idx >= len(self.layers):
            raise IndexError(
                f"HYV4 layer {layer_idx} is outside retained-state window "
                f"[{self.layer_offset}, {self.layer_offset + len(self.layers)})"
            )
        return local_idx

    def get_seq_length(self, layer_idx=0, cache_position=None):
        return self.layers[self._local_layer(layer_idx)].ckv.shape[-2]

    def get_query_offset(self, layer_idx=0):
        if self.position_ids is None:
            return 0
        return self.position_ids.min()

    def get_mask_sizes(self, query_length, layer_idx=0):
        # Mask construction may ask for layer 0 even when this window starts
        # later. Every HYV4 window state has the same fixed context shape.
        local_idx = 0 if int(layer_idx) < self.layer_offset else self._local_layer(layer_idx)
        return self.layers[local_idx].ckv.shape[-2], 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_kwargs = cache_kwargs or {"position_ids": self.position_ids}
        cache_kwargs = dict(cache_kwargs)
        cache_kwargs["position_ids"] = cache_kwargs["position_ids"].to(torch.int32)
        layer = self.layers[self._local_layer(layer_idx)]
        key_states = layer.update_ckv(key_states, cache_kwargs)
        value_states = layer.update_k_pe(value_states, cache_kwargs)
        return key_states, value_states

    def update_indexer(self, key_states, layer_idx):
        position_ids = self.position_ids.to(torch.int32)
        local_idx = self._local_layer(layer_idx)
        self.indexer_keys[local_idx] = ctx_scatter_3d(self.indexer_keys[local_idx], position_ids, key_states)
        return self.indexer_keys[local_idx]

    def touch_indexer(self, layer_idx):
        """Preserve a shared DSA state through an explicit state write.

        HF intentionally leaves the indexer cache unchanged on ``shared``
        layers.  A bare identity is numerically correct but SDK 1.22 cannot
        infer MDP ownership for an identity-only retained output.  Gathering
        and scattering the same rows is a mathematical no-op that keeps the
        exact retained-state ABI while giving the compiler a legal state
        producer.
        """
        position_ids = self.position_ids.to(torch.int32)
        local_idx = self._local_layer(layer_idx)
        state = self.indexer_keys[local_idx]
        updates = ctx_gather_3d(state, position_ids)
        self.indexer_keys[local_idx] = ctx_scatter_3d(state, position_ids, updates)
        return self.indexer_keys[local_idx]


class QEffHYV4Attention(HYV4Attention):
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                position_ids=None, prev_topk_indices=None, **kwargs):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        gate_states = self.gate_proj(hidden_states).view(batch_size, seq_length, -1, self.gate_projection_size)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        cache_kwargs = {"position_ids": position_ids, "batch_index": kwargs.get("batch_index")}
        if past_key_values is not None:
            k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx, cache_kwargs)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask[:, 0, :, :],
                position_ids,
                past_key_values=past_key_values,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            # Keep the shared layer's retained indexer state owned by a real
            # write in the ONNX graph; this is a semantic no-op.
            if past_key_values is not None and isinstance(past_key_values, QEffHYV4Cache):
                past_key_values.touch_indexer(self.layer_idx)
            topk_indices = prev_topk_indices

        num_classes = getattr(self, "_qeff_ctx_len", None)
        if num_classes is None:
            num_classes = key_states.shape[2]
        selected = F.one_hot(topk_indices.long(), num_classes=num_classes).sum(dim=-2).bool()
        index_mask = (~selected).unsqueeze(1)
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask & ~index_mask
        else:
            attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0,
            s_aux=self.sinks,
            **kwargs,
        )
        attn_output = attn_output * torch.sigmoid(gate_states)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        return self.o_proj(attn_output), attn_weights, topk_indices


class QEffHYV4MLP(HYV4MLP):
    pass


class QEffHYV4TopkRouter(HYV4TopkRouter):
    pass


class QEffHYV4Experts(HYV4Experts):
    """ONNX-safe top-k expert dispatch for HYV4.

    Transformers' optimized HYV4 expert interface defaults to grouped expert
    execution, which computes token counts with ``torch.histc``.  QAIC's ONNX
    exporter does not support that operator.  Dispatching each selected
    token/expert pair through batched gathers and BMMs preserves the HF
    routing and SwiGLU semantics without a loop over all 256 experts.
    """

    def _apply_gate(self, gate_up):
        """Apply HYV4 SwiGLU without exporting chunk as a sequence op.

        HYV4Experts._apply_gate uses Tensor.chunk. During layerwise export,
        the ONNX tracer lowers that form to SplitToSequence and SequenceAt,
        which QAIC does not import. The gate/up layout is fixed by
        moe_intermediate_size and the slices below are mathematically
        identical while remaining ordinary tensor operations.
        """
        split = self.intermediate_dim
        gate = gate_up[..., :split]
        up = gate_up[..., split:]
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        num_top_k = top_k_index.shape[-1]
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]

        token_idx = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1)
        token_idx = token_idx.expand(-1, num_top_k).reshape(-1)
        selected_hidden_states = hidden_states[token_idx]
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)

        selected_gate_up = self.gate_up_proj[expert_ids]
        proj_out = _batched_linear(
            selected_hidden_states,
            selected_gate_up,
            is_transposed=self.is_transposed,
        )
        proj_out = self._apply_gate(proj_out)

        selected_down = self.down_proj[expert_ids]
        proj_out = _batched_linear(
            proj_out,
            selected_down,
            is_transposed=self.is_transposed,
        )
        weighted_out = proj_out * sample_weights.unsqueeze(-1)
        final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        return final_hidden_states.to(hidden_states.dtype)


class QEffHYV4MoE(HYV4MoE):
    pass


class QEffHYV4HyperConnection(HYV4HyperConnection):
    pass


class QEffHYV4HyperHead(HYV4HyperHead):
    pass


class QEffHYV4DecoderLayer(HYV4DecoderLayer):
    pass


class QEffHYV4Model(HYV4Model):
    _start = 0
    _end = 0
    _total_layers = None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, prev_topk_indices=None, **kwargs):
        """Accept QEff's legacy cache tensors at the HF HYV4 boundary.

        QEff's exporter represents retained state as a list of ``(key, value)``
        tensors so it can name each tensor in the ONNX ABI.  Transformers 5.18
        HYV4 needs the richer ``DynamicCache`` interface because both MLA cache
        updates and the DSA indexer call cache methods such as
        ``get_query_offset`` and ``update_indexer``.  Convert only at this
        boundary; the underlying HF attention, MLA, and DSA implementations
        remain unchanged.
        """
        start, end = resolve_layer_window(QEffHYV4Model, len(self.layers))
        if past_key_values is not None and not isinstance(past_key_values, (Cache, QEffHYV4Cache)):
            past_key_values = QEffHYV4Cache.from_legacy_cache(
                self.config,
                past_key_values,
                layer_offset=start if is_layerwise_active() else 0,
            )
        if isinstance(past_key_values, QEffHYV4Cache):
            past_key_values.position_ids = position_ids
        if not is_layerwise_active():
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length(start) if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0) + past_seen_tokens
        if not isinstance(attention_mask, dict):
            attention_mask = {
                "indexed_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    allow_is_causal_skip=False,
                )
            }

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        hidden_states = (
            inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
            if inputs_embeds.ndim == 3
            else inputs_embeds
        )
        for layer_idx in range(start, end):
            hidden_states, prev_topk_indices = self.layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask["indexed_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        if is_last_layer_window(QEffHYV4Model, len(self.layers)):
            hidden_states = self.norm(self.hc_head(hidden_states))
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            attentions=prev_topk_indices,
        )


class QEffHYV4ForCausalLM(HYV4ForCausalLM):
    def set_qeff_export_context(self, ctx_len):
        """Propagate the fixed compile context to DSA attention modules."""
        if ctx_len is None:
            return
        self._qeff_export_ctx_len = int(ctx_len)
        for module in self.modules():
            if isinstance(module, QEffHYV4Attention):
                module._qeff_ctx_len = int(ctx_len)

    def get_qeff_export_seq_len(self, seq_len):
        """Keep the traced token sequence within the compiled cache context."""
        ctx_len = getattr(self, "_qeff_export_ctx_len", None)
        return min(seq_len, ctx_len) if ctx_len is not None else seq_len

    def get_onnx_past_key_value_names(self, layer_idx, layer_state=None):
        del layer_state
        return [f"past_key.{layer_idx}", f"past_value.{layer_idx}", f"indexer_key.{layer_idx}"]

    def get_onnx_past_key_value_dynamic_axes(self, state_idx, *, continuous_batching=False):
        batch_name = "full_batch_size" if continuous_batching else "batch_size"
        return {0: batch_name, 2: "ctx_len"} if state_idx < 2 else {0: batch_name, 1: "ctx_len"}

    def get_retained_state_names(self):
        return [
            name
            for layer_idx in range(self.config.num_hidden_layers)
            for name in (
                f"past_key.{layer_idx}",
                f"past_value.{layer_idx}",
                f"indexer_key.{layer_idx}",
            )
        ]

    def get_layerwise_input_tensors(self, example_inputs):
        start = int(getattr(self.model, "_start", 0))
        indexer_types = getattr(self.config, "indexer_types", None)
        if not is_layerwise_active() or start == 0 or not indexer_types or indexer_types[start] != "shared":
            return {}
        input_ids = example_inputs.get("input_ids", example_inputs["position_ids"])
        ctx_len = int(getattr(self, "_qeff_export_ctx_len", input_ids.shape[1]))
        topk = min(int(self.config.index_topk), ctx_len)
        return {"prev_topk_indices": torch.zeros((input_ids.shape[0], input_ids.shape[1], topk), dtype=torch.int32)}

    def get_layerwise_dynamic_axes(self, name):
        if name in {"prev_topk_indices", "topk_indices"}:
            return {0: "batch_size", 1: "seq_len", 2: "index_topk"}
        return {}

    def get_layerwise_ignored_inputs(self):
        return {"logits_to_keep"}

    def get_layerwise_static_inputs(self):
        start = int(getattr(self.model, "_start", 0))
        indexer_types = getattr(self.config, "indexer_types", None)
        return {"prev_topk_indices"} if indexer_types and indexer_types[start] == "shared" else set()

    def get_layerwise_output_names(self):
        return ["topk_indices"]

    def build_layerwise_inputs_embeds(self, token_ids):
        return torch.rand(
            token_ids.shape[0], token_ids.shape[1], self.config.hc_mult, self.config.hidden_size,
            dtype=next(self.parameters()).dtype,
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=1,
                prev_topk_indices=None, **kwargs):
        if is_layerwise_active():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                prev_topk_indices=prev_topk_indices, **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            if is_last_layer_window(QEffHYV4Model, len(self.model.layers)):
                logits = self.lm_head(hidden_states[:, slice(-1, None), :].to(dtype=self.lm_head.weight.dtype))
            else:
                logits = hidden_states
            outputs = CausalLMOutputWithPast(
                loss=None, logits=logits, past_key_values=outputs.past_key_values,
                hidden_states=None, attentions=outputs.attentions,
            )
        else:
            outputs = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, logits_to_keep=logits_to_keep, prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        if is_layerwise_active() and isinstance(outputs.past_key_values, QEffHYV4Cache):
            legacy_cache = tuple(
                (layer.ckv, layer.k_pe, indexer)
                for layer, indexer in zip(outputs.past_key_values.layers, outputs.past_key_values.indexer_keys)
            )
            return CausalLMOutputWithPast(
                loss=outputs.loss, logits=outputs.logits, past_key_values=legacy_cache,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )
        if torch.onnx.is_in_onnx_export() and isinstance(outputs.past_key_values, QEffHYV4Cache):
            legacy_cache = tuple(
                (layer.ckv, layer.k_pe, indexer)
                for layer, indexer in zip(outputs.past_key_values.layers, outputs.past_key_values.indexer_keys)
            )
            return CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=legacy_cache,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return outputs

    def get_dummy_pkv_cache(self, config, batch_size, seq_len, dtype=None):
        """Return the asymmetric compressed MLA cache expected by HYV4.

        HYV4 stores the non-RoPE compressed latent and the RoPE component as
        separate single-head cache tensors.  The generic QEff cache helper
        assumes a conventional full K/V cache, which gives the HF update path
        incompatible shapes during ONNX tracing.
        """
        dtype = dtype or getattr(config, "torch_dtype", torch.float32)
        seq_len = getattr(self, "_qeff_export_ctx_len", seq_len)
        cache = []
        for _ in range(config.num_hidden_layers):
            cache.append(
                [
                    torch.zeros((batch_size, 1, seq_len, config.kv_lora_rank), dtype=dtype),
                    torch.zeros((batch_size, 1, seq_len, config.qk_rope_head_dim), dtype=dtype),
                    torch.zeros((batch_size, seq_len, config.index_head_dim), dtype=dtype),
                ]
            )
        return cache

    def get_submodules_for_export(self):
        return {self.model.layers[0].__class__}
