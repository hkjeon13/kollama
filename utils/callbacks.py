from slacker import Slacker
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)

class SlackOnLogCallback(TrainerCallback):
    def __init__(
            self,
            slack_token: str,
            message_prefix: str,
            message_channel: str = "general",
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.slack_bot = Slacker(slack_token)
        self.message_prefix = message_prefix
        self.message_channel = message_channel
    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs=None,
            **kwargs
    ) -> None:
        if logs is not None:
            _message = ", ".join(": ".join(str(e) for e in item) for item in logs.items())

        self.slack_bot.chat.post_message(
            self.message_channel,
            self.message_prefix + _message
        )