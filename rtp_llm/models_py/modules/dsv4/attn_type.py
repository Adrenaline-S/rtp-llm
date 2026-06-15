"""DSV4 model-local KV-pool attn_type IDs and framework cache tags.

Lookup helpers that actually read ``attn_inputs`` / ``kv_cache`` live in
:mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.
"""

# Canonical DSV4-local attn_type ids. These are model metadata keys only;
# framework KVCache routing uses the tag strings below.
SWA_KV = 7
CSA_KV = 1
HCA_KV = 2
INDEXER_KV = 3
INDEXER_STATE = 4
CSA_STATE = 5
HCA_STATE = 6

ATTN_TYPE_TO_TAG = {
    CSA_KV: "csa_kv",
    HCA_KV: "hca_kv",
    INDEXER_KV: "indexer_kv",
    INDEXER_STATE: "indexer_state",
    CSA_STATE: "csa_state",
    HCA_STATE: "hca_state",
    SWA_KV: "swa_kv",
}

TAG_TO_ATTN_TYPE = {tag: attn_type for attn_type, tag in ATTN_TYPE_TO_TAG.items()}
