# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import pytest
import deepspeed
from deepspeed.ops.op_builder import OpBuilder
from unit.hpu import *
from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM)

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)


@pytest.mark.seq_inference
@pytest.mark.parametrize("batch_size", [1, 2], ids=["bsz=1", "bsz=2"])
@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-neo-1.3B", "facebook/opt-1.3b"])
class TestHybridEngineTextGen(DistributedTest):
    world_size = 1

    def _generate(self, model, tokenizer, prompt):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        tokens = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].to(f'{get_accelerator().device_name()}:{local_rank}')
        output = model.generate(**tokens, do_sample=False, max_length=100)
        outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
        return outputs

    def get_model(self, model_name):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.dropout = 0.0
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
        dev = get_accelerator().device_name()
        dtype = torch.float16
        if dev == 'hpu':
            if os.getenv("REPLACE_FP16", default=None):
                dtype = torch.bfloat16
        model = model.to(dtype=dtype)
        model = model.to(f'{dev}:{local_rank}')
        return model

    def get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_prompt(self, batch_size):
        if batch_size == 1:
            prompt = ["Microsoft is in Washington"]
        elif batch_size == 2:
            prompt = ["DeepSpeed is", "Microsoft is in Washington"]
        else:
            raise NotImplementedError(f"batch_size {batch_size} not implemented")
        return prompt

    def test_correctness(self, batch_size, model_name):
        pytest.skip("skip test for now, will fix in follow-up PR")
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        prompt = self.get_prompt(batch_size)

        base_out = self._generate(model, tokenizer, prompt)

        ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                ds_config["fp16"]["enabled"] = False
                ds_config["bf16"] = {"enabled": True}
            hpu_flag, msg = is_hpu_supported(ds_config)
            if not hpu_flag:
                pytest.skip(msg)

        model, *_ = deepspeed.initialize(model=model, config=ds_config)
        model.eval()
        ds1_out = self._generate(model, tokenizer, prompt)
        assert base_out == ds1_out, f"base_out: {base_out}, ds1_out: {ds1_out}"

        model.train()
        model.eval()
        ds2_out = self._generate(model, tokenizer, prompt)
        assert base_out == ds2_out

    def test_functionality(self, batch_size, model_name):
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        prompt = self.get_prompt(batch_size)

        ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                ds_config["fp16"]["enabled"] = False
                ds_config["bf16"] = {"enabled": True}
            hpu_flag, msg = is_hpu_supported(ds_config)
            if not hpu_flag:
                pytest.skip(msg)
        model, *_ = deepspeed.initialize(model=model, config=ds_config)

        model.eval()
        ds1_out = self._generate(model, tokenizer, prompt)

        model.train()
        model.eval()
        ds2_out = self._generate(model, tokenizer, prompt)

        assert ds1_out == ds2_out, f"ds1_out: {ds1_out}, ds2_out: {ds2_out}"
