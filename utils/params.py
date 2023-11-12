from dataclasses import dataclass, field


@dataclass
class LoraParams:
    apply_lora: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA"}
    )

    lora_r: int = field(
        default=64,
        metadata={"help": "The r value of LoRA"}
    )

    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha value of LoRA"}
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout value of LoRA"}
    )


@dataclass
class BnBParams:
    apply_4bit_training: bool = field(
        default=False,
        metadata={"help": "Whether to apply 4-bit training with bitsandbytes"}
    )

    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use double quantization for BnB 4bit"}
    )

    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "The quantization type for BnB 4bit"}
    )

    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "The compute dtype for BnB 4bit"}
    )


@dataclass
class SlackParams:
    use_slack_notifier: bool = field(
        default=False,
        metadata={"help": "Whether to use slack notifier"}
    )

    slack_token: str = field(
        default="",
        metadata={"help": "The slack token"}
    )

    slack_channel: str = field(
        default="general",
        metadata={"help": "The slack channel"}
    )

    slack_message_prefix: str = field(
        default="",
        metadata={"help": "The slack message prefix"}
    )
