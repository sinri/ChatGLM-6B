import os
import time
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


class ChatGLM6BWrapper:

    @staticmethod
    def load_tokenizer(model: str):
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    @staticmethod
    def load_config(model: str, pre_seq_len: int = 128):
        return AutoConfig.from_pretrained(model, trust_remote_code=True, pre_seq_len=pre_seq_len)

    @staticmethod
    def load_model(model: str, config: Optional = None):
        return AutoModel.from_pretrained(model, config=config, trust_remote_code=True).half().cuda().eval()

    @staticmethod
    def load(model: str, config: Optional = None):
        return ChatGLM6BWrapper(ChatGLM6BWrapper.load_tokenizer(model), ChatGLM6BWrapper.load_model(model, config))

    def __init__(self, tokenizer, model):
        self.__tokenizer = tokenizer
        self.__model = model
        self.__history: List[Tuple[str, str]] = []

    def load_extra_ckpt(self, extra_ckpt_dir: str):
        prefix_state_dict = torch.load(os.path.join(extra_ckpt_dir, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        self.__model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        return self

    def clean_history(self):
        self.__history = []
        return self

    def chat(
            self,
            query: str, max_length: int = 2048, num_beams=1,
            do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs
    ):
        response, history = self.__model.chat(
            self.__tokenizer,
            query,
            history=self.__history,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            logits_processor=logits_processor,
            **kwargs
        )
        self.__history = history
        return response

    def start_in_cli(self,keep_history:bool):
        while True:
            query = input("\n> ")
            if query.strip() == '':
                break

            print(f'[{time.time()}] request')
            r = self.chat(query)
            print(f'[{time.time()}] responded')
            print(f'< {r}')

            if not keep_history:
                self.clean_history()


if __name__ == '__main__':
    model_path = 'E:\\OneDrive\\Leqee\\ai\\repo\\THUDM\\chatglm-6b'
    c = ChatGLM6BWrapper.load(model_path)
    c.start_in_cli(True)
