# from peft import PeftConfig, PeftModel
# from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from typing import Literal, Optional
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from dsp.modules.lm import LM
from transformers import BitsAndBytesConfig

# from dsp.modules.finetuning.finetune_hf import preprocess_prompt


def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class HFModelCC(LM):
    def __init__(
        self,
        model: str,
        checkpoint: Optional[str] = None,
        is_client: bool = False,
        hf_device_map: Literal[
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        ] = "auto",
        token: Optional[str] = None,
    ):
        """wrapper for Hugging Face models

        Args:
            model (str): HF model identifier to load and use
            checkpoint (str, optional): load specific checkpoints of the model. Defaults to None.
            is_client (bool, optional): whether to access models via client. Defaults to False.
            hf_device_map (str, optional): HF config strategy to load the model.
                Recommeded to use "auto", which will help loading large models using accelerate. Defaults to "auto".
        """

        super().__init__(model)
        self.provider = "hf"
        self.is_client = is_client
        self.device_map = hf_device_map

        hf_autoconfig_kwargs = dict(token=token or os.environ.get("HF_TOKEN"))
        hf_autotokenizer_kwargs = hf_autoconfig_kwargs.copy()
        hf_automodel_kwargs = hf_autoconfig_kwargs.copy()
        self.device = 'cuda'
        model_name = model
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=False)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        access_token = "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"
        model_name = 'meta-llama/Llama-2-13b-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=bnb_config , token=access_token)


        self.rationale = True
        self.drop_prompt_from_output = True   
        self.history = []

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def _generate(self, prompt, **kwargs):
        assert not self.is_client
        # TODO: Add caching
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        # print(prompt)
        if isinstance(prompt, dict):
            try:
                prompt = prompt["messages"][0]["content"]
            except (KeyError, IndexError, TypeError):
                print("Failed to extract 'content' from the prompt.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # print(kwargs)
        outputs = self.model.generate(**inputs, **kwargs)
        if self.drop_prompt_from_output:
            input_length = inputs.input_ids.shape[1]
            outputs = outputs[:, input_length:]
        completions = [{"text": c} for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]


# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
# @NotebookCacheMemory.cache
# def cached_generate(self, prompt, **kwargs):
#      return self._generate(prompt, **kwargs)