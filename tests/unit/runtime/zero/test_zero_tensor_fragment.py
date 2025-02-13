# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch
import os

from unit.common import DistributedTest
from unit.simple_model import random_dataloader, SimpleModel
from unit.util import bf16_required_version_check

import deepspeed
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from deepspeed.utils import safe_set_full_fp32_param, safe_set_full_optimizer_state
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.aio import AsyncIOBuilder
from unit.hpu import *

WEIGHT_KEY = 'weight'
FIRST_ORDER_KEY = 'exp_avg'
SECOND_ORDER_KEY = 'exp_avg_sq'


def validate_full_tensors(model):
    for _, lp in model.named_parameters():
        hp = safe_get_full_fp32_param(lp)
        exp_avg = safe_get_full_optimizer_state(lp, 'exp_avg')
        exp_avg_sq = safe_get_full_optimizer_state(lp, 'exp_avg_sq')
        hp_grad = safe_get_full_grad(lp)
        param_list = [hp, hp_grad, exp_avg, exp_avg_sq]
        if lp.requires_grad:
            assert all([p is not None for p in param_list])
        else:
            assert all([p is None for p in param_list])


class MyModel(torch.nn.Module):

    def __init__(self, hidden_dim, frozen_weights):
        super(MyModel, self).__init__()
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, 1),
             torch.nn.Linear(1, 1),
             torch.nn.Linear(1, hidden_dim)])
        if frozen_weights:
            self.linears[0].weight.requires_grad = False
            self.linears[0].bias.requires_grad = False

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
            x = self.act(x)
        loss = self.cel(x, y)
        val = (x, loss)
        return val


def run_fragmented_model(model, config_dict, hidden_dim, dtype):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()
    for n, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])
        loss = loss[1]
        model.backward(loss)
        validate_full_tensors(model)
        model.step()

    # Needed in ZeRO 3. Not doing so can give memory leak
    model.destroy()


@pytest.mark.parametrize('frozen_weights', [True, False])
class TestTensorFragmentGet(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(self, tmpdir, zero_stage, offload_device, frozen_weights):
        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip('Skip tests since async-io is not compatible')

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 2
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        dtype = torch.float16
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["bf16"] = {"enabled": True}
                dtype = torch.bfloat16
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }

        hidden_dim = 128
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict, dtype=dtype):
                model = MyModel(hidden_dim, frozen_weights)
        else:
            model = MyModel(hidden_dim, frozen_weights)

        run_fragmented_model(model, config_dict, hidden_dim, dtype)

    def test_bf16_fragments(self, frozen_weights):
        if frozen_weights:
            pytest.skip("TODO: Frozen weights not currently supported by BF16 Optimizer")

        if (bool(pytest.use_hpu) != True) and (not bf16_required_version_check(accelerator_check=False)):
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 0,
            }
        }

        hidden_dim = 128
        model = MyModel(hidden_dim, frozen_weights)
        run_fragmented_model(model, config_dict, hidden_dim, torch.bfloat16)


def create_random_values(model, key_list, group):
    param_values = {}
    for n, lp in model.named_parameters():
        param_shape = lp.ds_shape if hasattr(lp, 'ds_id') else lp.shape
        param_values[n] = {}
        for key in key_list:
            rand_value = torch.rand(param_shape, dtype=torch.float32, device=model.device)
            dist.broadcast(rand_value, src=0, group=group)
            param_values[n][key] = rand_value
    return param_values


def set_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, value_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                safe_set_full_fp32_param(lp, value_tensor)
            else:
                safe_set_full_optimizer_state(lp, value_tensor, key)


def validate_param_values_with_dict(model, value_dict):
    for n, lp in model.named_parameters():
        for key, expected_tensor in value_dict[n].items():
            if key == WEIGHT_KEY:
                actual_tensor = safe_get_full_fp32_param(lp)
            else:
                actual_tensor = safe_get_full_optimizer_state(lp, key)
            assert torch.equal(expected_tensor, actual_tensor)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
class TestTensorFragmentUpdate(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(self, tmpdir, zero_stage, offload_device, dtype):

        if dtype == torch.bfloat16 and not bf16_required_version_check(accelerator_check=False):
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip('Skip tests since async-io is not compatible')

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }

        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        hidden_dim = 128
        if zero_stage == 3:
            config_dict["zero_optimization"]["param_persistence_threshold"] = hidden_dim
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = SimpleModel(hidden_dim, nlayers=4)
        else:
            model = SimpleModel(hidden_dim, nlayers=4)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        world = dist.get_world_size()
        group = dist.new_group(ranks=list(range(world)))

        dist.barrier()
        optim_keys = [WEIGHT_KEY, FIRST_ORDER_KEY, SECOND_ORDER_KEY]
        optim_state_values = create_random_values(model, optim_keys, group)
        set_param_values_with_dict(model, optim_state_values)
        validate_param_values_with_dict(model, optim_state_values)

        # Needed in ZeRO 3. Not doing so can leak memory.
        model.destroy()
