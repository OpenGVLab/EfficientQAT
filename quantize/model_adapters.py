"""Per-architecture glue for block-wise QAT (Block-AP).

block_ap.py is model-agnostic; everything that differs between model families lives
in an adapter here. To support a new model: subclass BlockAPAdapter, implement the
handful of methods below, and register it in get_adapter().

A decoder block may need, beyond its hidden state:
  - static kwargs    : sample-independent given a fixed seqlen (attention masks,
                       position_ids, position_embeddings). Captured ONCE per layer.
  - per_layer_input  : per-sample, derived from input_ids, INDEPENDENT of the
                       quantized weights (e.g. Gemma PLE). Recomputed per batch.
  - shared_kv        : per-sample K/V produced by one block and reused by later
                       blocks (Gemma KV-sharing). Propagated like hidden states.

Adapters expose these so block_ap.py can capture/propagate them uniformly.
"""
from collections import UserDict
import torch


def _detach(x):
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, (tuple, list)):
        return type(x)(_detach(v) for v in x)
    return x


class BlockAPAdapter:
    """Default: a plain Llama-style decoder stack at model.model.layers."""
    name = "default"

    def __init__(self, model):
        self.model = model
        self.static = None  # list[dict]: per-layer sample-independent kwargs

    # --- structure -----------------------------------------------------------
    @property
    def _root(self):
        return self.model.model

    @property
    def layers(self):
        return self._root.layers

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    def pre_layer_module_names(self):
        names = ["embed_tokens", "norm"]
        if hasattr(self._root, "rotary_emb"):
            names.append("rotary_emb")
        return names

    def move_pre_layers(self, dev):
        for n in self.pre_layer_module_names():
            setattr(self._root, n, getattr(self._root, n).to(dev))

    def final_norm(self):
        return self._root.norm

    def forward_for_capture(self, input_ids):
        """Full-depth forward used only to capture static kwargs (layers are
        identity-patched by capture_static_kwargs, so no block compute happens)."""
        return self._root(input_ids, use_cache=False)

    # --- static kwarg capture ------------------------------------------------
    def capture_static_kwargs(self, run_one_forward):
        """Record per-layer sample-independent kwargs (masks, position_ids,
        position_embeddings) in ONE forward. Each layer is replaced by an identity
        recorder that returns its hidden state unchanged — masks/rope don't depend
        on hidden values, so this runs full depth cheaply WITHOUT every layer on the
        GPU. `run_one_forward` runs the model on one calib batch."""
        layers = self.layers
        static = [None] * len(layers)
        orig = [l.forward for l in layers]

        def mk(i):
            def g(hidden_states, *a, **kw):
                static[i] = self._extract_static(a, kw)
                return hidden_states  # identity: we only want the kwargs
            return g

        for i, l in enumerate(layers):
            l.forward = mk(i)
        try:
            run_one_forward()
        except Exception:
            pass  # identity outputs may break downstream model code; kwargs are already recorded
        finally:
            for i, l in enumerate(layers):
                l.forward = orig[i]
        missing = [i for i, s in enumerate(static) if s is None]
        assert not missing, f"static kwargs not captured for layers {missing}"
        self.static = static
        return static

    def _extract_static(self, args, kwargs):
        out = {}
        for k in ("attention_mask", "position_ids", "position_embeddings"):
            if kwargs.get(k) is not None:
                out[k] = _detach(kwargs[k])
        return out

    # --- per-block forward ---------------------------------------------------
    def needs_per_layer_input(self):
        return False

    def per_layer_input(self, i, input_ids):
        return None

    def is_kv_producer(self, i):
        """Return a layer_type key if block i produces shared KV, else None."""
        return None

    def is_kv_consumer(self, i):
        return False

    def run_block(self, i, layer, hidden, per_layer_input=None, shared_kv=None,
                  produce_shared_kv=False):
        """Run block i. Returns (hidden_out, produced_shared_kv_or_None)."""
        kw = dict(self.static[i]) if self.static and self.static[i] else {}
        out = layer(hidden, **kw)
        return (out[0] if isinstance(out, tuple) else out), None


class Gemma4Adapter(BlockAPAdapter):
    """Gemma-4 (E*B / Gemma-3n lineage): multimodal wrapper, PLE, KV-sharing,
    alternating sliding/global attention."""
    name = "gemma4"

    def __init__(self, model):
        super().__init__(model)
        self._ple_cache = {}  # id(input_ids) -> full [B,T,L,256] PLE (same across all blocks)
        cfg = model.config.get_text_config()
        self.layer_types = cfg.layer_types
        n = cfg.num_hidden_layers
        shared = getattr(cfg, "num_kv_shared_layers", 0)
        self.first_shared = n - shared if shared else n
        # producer = last non-shared layer of each type (it writes shared_kv_states[type])
        last_of_type = {}
        for i in range(self.first_shared):
            last_of_type[self.layer_types[i]] = i
        self.producers = {i: t for t, i in last_of_type.items()}

    @property
    def _root(self):
        return self.model.model.language_model

    @property
    def hidden_size(self):
        return self.model.config.get_text_config().hidden_size

    def pre_layer_module_names(self):
        names = ["embed_tokens", "norm"]
        for extra in ("rotary_emb", "rotary_emb_local", "embed_tokens_per_layer",
                      "per_layer_model_projection", "per_layer_projection_norm",
                      "altup_projections", "altup_unembed_projections"):
            if hasattr(self._root, extra):
                names.append(extra)
        return names

    def needs_per_layer_input(self):
        return True

    def per_layer_input(self, i, input_ids):
        # Mirror Gemma4TextModel.forward: embed -> get_per_layer_inputs -> project, slice block i.
        # The full [B,T,L,256] is identical for every block of a given batch (only the slice
        # differs) and is independent of the quantized weights, so cache it per batch — without
        # this the projection reruns 35x per batch per pass and dominates runtime.
        # ponytail: cache holds all batches (~5GB CPU RAM at train_size 256/seqlen 1024); if RAM
        # is tight, drop train_size rather than the cache (recompute is hours of CPU).
        key = id(input_ids)
        full = self._ple_cache.get(key)
        if full is None:
            r = self._root
            embeds = r.embed_tokens(input_ids)
            full = r.project_per_layer_inputs(embeds, r.get_per_layer_inputs(input_ids, embeds))
            self._ple_cache[key] = full
        return full[:, :, i, :]

    def is_kv_producer(self, i):
        return self.producers.get(i, None)

    def is_kv_consumer(self, i):
        return i >= self.first_shared

    def run_block(self, i, layer, hidden, per_layer_input=None, shared_kv=None,
                  produce_shared_kv=False):
        kw = dict(self.static[i]) if self.static and self.static[i] else {}
        produced = None
        if self.is_kv_producer(i) is not None:
            # producer writes its K/V into this dict in place (unconditionally), so it
            # must always be supplied — even on training/fp passes where we don't keep it.
            produced = UserDict()
            kw["shared_kv_states"] = produced
            if produce_shared_kv:
                kw["return_shared_kv_states"] = True
        elif self.is_kv_consumer(i) and shared_kv is not None:
            kw["shared_kv_states"] = shared_kv
        out = layer(hidden, per_layer_input, **kw)
        hid = out[0] if isinstance(out, tuple) else out
        return hid, (produced if produce_shared_kv else None)


def get_adapter(model):
    if model.__class__.__name__.startswith("Gemma4"):
        return Gemma4Adapter(model)
    return BlockAPAdapter(model)
