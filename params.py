from dataclasses import dataclass, field

@dataclass
class LlamaParams:
    vocab_size: int = field(
        default=32000,
        metadata={"help": "The size of vocabulary"}
    )
    hidden_size: int = field(
        default=4096,
        metadata={"help": "Size of the hidden layers"}
    )
    intermediate_size: int = field(
        default=11008,
        metadata={"help": "Size of the intermediate (feed-forward) layer"}
    )
    num_hidden_layers: int = field(
        default=16,
        metadata={"help": "Number of hidden layers"}
    )
    num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads"}
    )
    num_key_value_heads: int = field(
        default=16,
        metadata={"help": "Number of key/value heads"}
    )
    hidden_act: str = field(
        default="silu",
        metadata={"help": "Activation function for hidden layers"}
    )
    max_position_embeddings: int = field(
        default=4096,
        metadata={"help": "Maximum number of position embeddings"}
    )
    initializer_range: float = field(
        default=0.02,
        metadata={"help": "Range of the initializer"}
    )
    rms_norm_eps: float = field(
        default=1e-06,
        metadata={"help": "Epsilon for RMS normalization"}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use caching"}
    )
    pad_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "Padding token ID"}
    )
    bos_token_id: int = field(
        default=1,
        metadata={"help": "Beginning of sentence token ID"}
    )
    eos_token_id: int = field(
        default=2,
        metadata={"help": "End of sentence token ID"}
    )
    pretraining_tp: int = field(
        default=1,
        metadata={"help": "Whether to use tensor parallelism during pretraining"}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to tie word embeddings"}
    )
    rope_theta: float = field(
        default=10000.0,
        metadata={"help": "Theta parameter for ROPE"}
    )
    rope_scaling: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use ROPE scaling"}
    )
    attention_bias: bool = field(
        default=False,
        metadata={"help": "Whether to use attention bias"}
    )
