# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import torch.nn as nn
import pytest
import os
import deepspeed.comm as dist
from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from deepspeed.runtime.pipe.module import PipelineModule
from unit.alexnet_model import AlexNetPipe, train_cifar
from unit.common import DistributedTest
from unit.util import skip_on_arch
from unit.hpu import *

PipeTopo = PipeDataParallelTopology

config_dict = {
    "train_batch_size": 4,
    "grandient_accumulation_steps": 1,
    "steps_per_print": 20,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "zero_optimization": {
        "stage": 0
    },
    "fp16": {
        "enabled": False
    },
    "pipeline": {
        "seed_layers": True,
        "activation_checkpoint_interval": 1
    }
}


def rel_diff(A, B):
    return abs(A - B) / abs(A)


@pytest.mark.parametrize('topo_config', [
    {
        "num_pp": 1,
        "num_dp": 4
    },
    {
        "num_pp": 2,
        "num_dp": 2
    },
    {
        "num_pp": 4,
        "num_dp": 1
    },
])
class TestPipeCifar10(DistributedTest):
    world_size = 4

    def test_pipe_base(self, topo_config):
        skip_on_arch(min_arch=7)
        if bool(pytest.use_hpu) == True:
            if get_hpu_dev_version() == "Gaudi":
                os.environ['PT_ENABLE_COMM_GROUP_CACHE'] = "true"
        topo = PipeTopo(**topo_config)
        steps = 100  # must be >=100
        # set random seed to make sure the weights in all ranks are the same for both dp/pp
        import deepspeed.runtime.utils as ds_utils
        ds_utils.set_random_seed(0)
        # Allocate model for consistent initial weights.
        init_net = AlexNetPipe()

        base_net = copy.deepcopy(init_net)
        base_model = PipelineModule(layers=base_net.to_layers(), num_stages=1, loss_fn=nn.CrossEntropyLoss())

        # Train with just data parallelism
        base_losses = train_cifar(base_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])

        test_net = copy.deepcopy(init_net)
        test_model = PipelineModule(layers=test_net.to_layers(),
                                    topology=topo,
                                    loss_fn=nn.CrossEntropyLoss(),
                                    use_hpu=(bool(pytest.use_hpu) == True))

        test_losses = train_cifar(test_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])

        abs_diffs = [l0 - l1 for l0, l1 in zip(base_losses, test_losses)]
        rel_diffs = [rel_diff(l0, l1) for l0, l1 in zip(base_losses, test_losses)]
        if dist.get_rank() == 0:
            print(f'abs min={min(abs_diffs)} max={max(abs_diffs)} avg={sum(abs_diffs)/len(abs_diffs)}')
            print(f'rel min={min(rel_diffs)} max={max(rel_diffs)} avg={sum(rel_diffs)/len(rel_diffs)}')
            print(f'first: base={base_losses[0]} test={test_losses[0]} abs={abs_diffs[0]} rel={rel_diffs[0]}')

            for lastX in [1, 10, 100]:
                base_avg = sum(base_losses[-lastX:]) / lastX
                test_avg = sum(test_losses[-lastX:]) / lastX
                print(
                    f'last-{lastX}: base={base_avg} test={test_avg} abs={base_avg - test_avg} rel={rel_diff(base_avg, test_avg)}'
                )

        lastX = 100
        base = base_losses[-lastX:]
        base_avg = sum(base) / len(base)
        test = test_losses[-lastX:]
        test_avg = sum(test) / len(test)
        assert rel_diff(base_avg, test_avg) < 0.05  # Originally 0.03, but seeing instability with AMD results

    # def _check_model_params_equal(self, model1, model2):
    #     for p1, p2 in zip(model1.parameters(), model2.parameters()):
    #         if p1.data.ne(p2.data).sum() > 0:
    #             assert False, f"model params not equal"

    def test_pipe_use_reentrant(self, topo_config):
        skip_on_arch(min_arch=7)

        if bool(pytest.use_hpu) == True:
            if get_hpu_dev_version() == "Gaudi":
                os.environ['PT_ENABLE_COMM_GROUP_CACHE'] = "true"
        topo = PipeTopo(**topo_config)
        steps = 100  # must be >=100
        # set random seed to make sure the weights in all ranks are the same for both dp/pp
        import deepspeed.runtime.utils as ds_utils
        ds_utils.set_random_seed(0)

        # Allocate model for consistent initial weights.
        init_net = AlexNetPipe()

        # Train with not set use_reentrant, default: True
        base_net = copy.deepcopy(init_net)
        base_model = PipelineModule(layers=base_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        base_losses = train_cifar(base_model, config=config_dict, num_steps=steps, fp16=config_dict['fp16']['enabled'])

        # Train with set use_reentrant=False, this will use ``non_reentrant_checkpoint``
        test_config_dict = copy.deepcopy(config_dict)
        test_config_dict['pipeline']['use_reentrant'] = False
        test_net = copy.deepcopy(init_net)
        test_model = PipelineModule(layers=test_net.to_layers(), topology=topo, loss_fn=nn.CrossEntropyLoss())
        test_losses = train_cifar(test_model,
                                  config=test_config_dict,
                                  num_steps=steps,
                                  fp16=config_dict['fp16']['enabled'])

        abs_diffs = [l0 - l1 for l0, l1 in zip(base_losses, test_losses)]
        rel_diffs = [rel_diff(l0, l1) for l0, l1 in zip(base_losses, test_losses)]
        if dist.get_rank() == 0:
            print(f'abs min={min(abs_diffs)} max={max(abs_diffs)} avg={sum(abs_diffs)/len(abs_diffs)}')
            print(f'rel min={min(rel_diffs)} max={max(rel_diffs)} avg={sum(rel_diffs)/len(rel_diffs)}')
            print(f'first: base={base_losses[0]} test={test_losses[0]} abs={abs_diffs[0]} rel={rel_diffs[0]}')

            for lastX in [1, 10, 100]:
                base_avg = sum(base_losses[-lastX:]) / lastX
                test_avg = sum(test_losses[-lastX:]) / lastX
                print(
                    f'last-{lastX}: base={base_avg} test={test_avg} abs={base_avg - test_avg} rel={rel_diff(base_avg, test_avg)}'
                )
        lastX = 100
        base = base_losses[-lastX:]
        base_avg = sum(base) / len(base)
        test = test_losses[-lastX:]
        test_avg = sum(test) / len(test)
        assert rel_diff(base_avg, test_avg) < 0.05

        # the following check could passed on higher version docker: nvcr.io/nvidia/pytorch:23.07-py3(torch2.1.0 cuda12.1)
        # Check if models have same weights after training
        # self._check_model_params_equal(base_model, test_model)
