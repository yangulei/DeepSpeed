# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from types import SimpleNamespace
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save, get_model_ckpt_name_for_rank
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version

from unit.common import DistributedTest, DistributedFixture
from unit.simple_model import *
from unit.hpu import *

from unit.checkpoint.common import *

import pytest


class TestZeROCheckpoint(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [3])
    def test_pipeline_checkpoint_loading(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
                "pipeline_loading_checkpoint": True,
            }
        }
        hidden_dim = 10

        with deepspeed.zero.Init():
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True)

    @pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer', [(1, False, 'Adam'), (2, False, 'Adam'),
                                                                             (2, True, 'deepspeed_adam'),
                                                                             (3, False, 'Adam'),
                                                                             (3, True, 'deepspeed_adam')])
    def test_load_optimizer_state(self, tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "wall_clock_breakdown": True,
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }
        hidden_dim = 10
        fp16 = True
        zero3_init_dtype = None
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                fp16 = False
                zero3_init_dtype = torch.float32
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        if zero_stage == 3:
            with deepspeed.zero.Init(dtype=zero3_init_dtype):
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            fp16=fp16,
                                            load_optimizer_states=True)

    @pytest.mark.parametrize('zero_stage, use_cpu_offload, adam_optimizer', [(1, False, "Adam"), (2, False, "Adam"),
                                                                             (2, True, 'deepspeed_adam'),
                                                                             (3, False, 'Adam'),
                                                                             (3, True, 'deepspeed_adam')])
    def test_not_load_optimizer_state(self, tmpdir, zero_stage, use_cpu_offload, adam_optimizer):
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }
        hidden_dim = 10
        fp16 = True
        zero3_init_dtype = None
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                fp16 = False
                zero3_init_dtype = torch.float32
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        if zero_stage == 3:
            global DeepSpeedZeroOptimizer_Stage3
            from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
            with deepspeed.zero.Init(dtype=zero3_init_dtype):
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            fp16=fp16,
                                            load_optimizer_states=False)

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_hybrid_optimizer_state(self, tmpdir, zero_stage):
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "zero_allow_untested_optimizer": True,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 10
        fp16 = True
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                fp16 = False
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
        optimizers = [HybridStateOptimizer(model.parameters()) for model in models]

        checkpoint_correctness_verification(config_dict,
                                            models=models,
                                            base_optimizers=optimizers,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            fp16=fp16,
                                            load_optimizer_states=True)

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_load_module_only(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        hidden_dim = 10
        fp16 = True
        zero3_init_dtype = None
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                fp16 = False
                zero3_init_dtype = torch.float32
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        if zero_stage == 3:
            with deepspeed.zero.Init(dtype=zero3_init_dtype):
                models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]
        else:
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, fp16=fp16, load_module_only=True)


class ws4_model_checkpoint(DistributedFixture):
    world_size = 4

    def run(self, class_tmpdir, elastic_save, load_optim):
        ds_config = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 2,
                "elastic_checkpoint": elastic_save
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                ds_config["fp16"]["enabled"] = False
                ds_config["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    ds_config["communication_data_type"] = 'bfp16'
                dtype = torch.float
            hpu_flag, msg = is_hpu_supported(ds_config)
            if not hpu_flag:
                pytest.skip(msg)
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        if load_optim:
            torch.save(model.optimizer.optimizer.state_dict(), os.path.join(class_tmpdir, 'opt-state-dict'))
        model.save_checkpoint(class_tmpdir)


@pytest.mark.parametrize("elastic_save", [True, False])
@pytest.mark.parametrize("elastic_load", [True, False])
@pytest.mark.parametrize("load_optim", [True, False])
class TestZeROElasticCheckpoint(DistributedTest):
    world_size = 2

    def test_elastic_checkpoint_fixed_dp(self, tmpdir, elastic_save, elastic_load, load_optim):
        ds_config = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 2,
                "elastic_checkpoint": elastic_save
            }
        }
        hidden_dim = 10

        # torch 1.2.* stores raw tensor id numbers in checkpoint state which leads to
        # false positive mismatches in checkpoint state comparisons.
        # Newer torch versions store tensor ids as 0, 1, 2, ...
        expected_mismatch_keys = [] if required_torch_version(min_version=1.4) else ['params']
        models = [SimpleModel(hidden_dim) for _ in range(2)]

        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                ds_config["fp16"]["enabled"] = False
                ds_config["fp32"] = {"enabled": True}
                dtype = torch.float
                if get_hpu_dev_version() == "Gaudi":
                    ds_config["communication_data_type"] = 'bfp16'
            hpu_flag, msg = is_hpu_supported(ds_config)
            if not hpu_flag:
                pytest.skip(msg)
        model, _, _, _ = deepspeed.initialize(config=ds_config,
                                              model=models[0],
                                              model_parameters=models[0].parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        if load_optim:
            torch.save(model.optimizer.optimizer.state_dict(), os.path.join(tmpdir, 'opt-state-dict'))
        model.save_checkpoint(tmpdir)

        ds_config["zero_optimization"]["elastic_checkpoint"] = elastic_load
        model, _, _, _ = deepspeed.initialize(config=ds_config,
                                              model=models[1],
                                              model_parameters=models[1].parameters())
        model.load_checkpoint(tmpdir, load_optimizer_states=load_optim)

        if load_optim:
            saved_sd = torch.load(os.path.join(tmpdir, 'opt-state-dict'))
            curr_sd = model.optimizer.optimizer.state_dict()
            for curr_param_group, saved_param_group in zip(curr_sd['param_groups'], saved_sd['param_groups']):
                compare_state_dicts(curr_param_group, saved_param_group, expected_mismatch_keys)

        data_loader = random_dataloader(model=model,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_elastic_checkpoint_change_dp(self, ws4_model_checkpoint, class_tmpdir, elastic_save, elastic_load,
                                          load_optim):
        ds_config = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 2,
                "elastic_checkpoint": elastic_load
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                ds_config["fp16"]["enabled"] = False
                ds_config["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    ds_config["communication_data_type"] = 'bfp16'

            hpu_flag, msg = is_hpu_supported(ds_config)
            if not hpu_flag:
                pytest.skip(msg)
        # Load checkpoint with dp world size = 2
        model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
        if load_optim:
            with pytest.raises(deepspeed.runtime.zero.utils.ZeRORuntimeException):
                model.load_checkpoint(class_tmpdir, load_optimizer_states=load_optim)
        else:
            model.load_checkpoint(class_tmpdir, load_optimizer_states=load_optim)


class TestZeROSaveLoadEdgeCase(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_immediate_save_load(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        ds_model.save_checkpoint(tmpdir)
        ds_model.load_checkpoint(tmpdir,
                                 load_optimizer_states=False,
                                 load_lr_scheduler_states=False,
                                 load_module_only=False)

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_load_immediate_save(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 4,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        # 1. pretrain a model and save it
        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                dtype = torch.float
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        data_loader = random_dataloader(model=ds_model,
                                        total_samples=1,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=dtype)
        for _, batch in enumerate(data_loader):
            loss = ds_model(batch[0], batch[1])
            ds_model.backward(loss)
            ds_model.step()

        ds_model.empty_partition_cache()
        ds_model.save_checkpoint(tmpdir)

        # 2. load and immediately save a model with a fresh ds engine
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)
        ds_model.load_checkpoint(tmpdir,
                                 load_optimizer_states=False,
                                 load_lr_scheduler_states=False,
                                 load_module_only=False)
        ds_model.save_checkpoint(tmpdir)

    @pytest.mark.parametrize('zero_stage', [0, 1, 2, 3])
    def test_save_before_accum_grad_is_done(self, tmpdir, zero_stage):
        config_dict = {
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
                "stage3_gather_fp16_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": 2,
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 4,
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)

        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                dtype = torch.float
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)
        # This test reproduces a bug where one tries to retrieve a 16bit model before grad_accum
        # cycle was completed.
        # So we config grad_accum=2 and step only once and save_16bit_model
        ds_model = create_deepspeed_model(config_dict=config_dict, model=model, base_optimizer=None)

        data_loader = random_dataloader(model=ds_model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=ds_model.device,
                                        dtype=dtype)

        batch = next(iter(data_loader))
        loss = ds_model(batch[0], batch[1])
        ds_model.backward(loss)
        ds_model.step()

        ds_model.empty_partition_cache()

        # we stepped only once, and now save 16bit model before gradient_accumulation_steps=2 is complete
        ds_model.save_16bit_model(tmpdir, "model.pt")

        # let's test just as well that we can save the checkpoint too
        ds_model.save_checkpoint(tmpdir)


class TestZeROCheckpointFrozenWeights(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_load_optimizer_state(self, tmpdir, zero_stage):

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "wall_clock_breakdown": True,
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        fp16 = True
        dtype = None
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                fp16 = False
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
                dtype = torch.float32
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        hidden_dim = 10

        with deepspeed.zero.Init(enabled=zero_stage == 3, dtype=dtype):
            models = [SimpleFrozenModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_optimizer_states=True,
                                            fp16=fp16)

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_not_load_optimizer_state(self, tmpdir, zero_stage):

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 10
        fp16 = True
        dtype = None
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                fp16 = False
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                dtype = torch.float32
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)
        with deepspeed.zero.Init(enabled=zero_stage == 3, dtype=dtype):
            models = [SimpleFrozenModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict,
                                            models,
                                            hidden_dim,
                                            tmpdir,
                                            load_optimizer_states=False,
                                            fp16=fp16)

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_load_module_only(self, tmpdir, zero_stage):
        config_dict = {
            "train_batch_size": 2,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        hidden_dim = 10
        fp16 = True
        dtype = None
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                fp16 = False
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                dtype = torch.float32
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)
        with deepspeed.zero.Init(enabled=zero_stage == 3, dtype=dtype):
            models = [SimpleFrozenModel(hidden_dim, empty_grad=False) for _ in range(2)]

        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True, fp16=fp16)

    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_save_exclude_frozen_weights(self, tmpdir, zero_stage):
        world_size = 1
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": 'Adam'
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        hidden_dim = 10

        model = SimpleFrozenModel(hidden_dim, empty_grad=False)

        ds_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        # Validate backwards-compatibility of including frozen parameters in checkpoint
        all_ckpt_folder = os.path.join(tmpdir, 'all_params')
        ds_engine.save_checkpoint(all_ckpt_folder)
        all_params_ckpt_file = get_model_ckpt_name_for_rank(os.path.join(all_ckpt_folder, 'global_step0'), '00')
        loaded_all_param_model = torch.load(all_params_ckpt_file)['module']
        all_param_names = set([n for n, p in model.named_parameters()])
        assert set(loaded_all_param_model.keys()) == all_param_names

        # Validate exclusion of frozen parameters
        trainable_ckpt_folder = os.path.join(tmpdir, 'no_frozen_params')
        ds_engine.save_checkpoint(trainable_ckpt_folder, exclude_frozen_parameters=True)

        trainable_ckpt_file = get_model_ckpt_name_for_rank(os.path.join(trainable_ckpt_folder, 'global_step0'), '00')

        # Excluding frozen parameters should reduce checkpoint size
        assert os.path.getsize(all_params_ckpt_file) > os.path.getsize(trainable_ckpt_file)

        loaded_trainable_param_model = torch.load(trainable_ckpt_file)['module']
        frozen_param_names = set([n for n, p in model.named_parameters() if not p.requires_grad])
        loaded_trainable_param_names = set(loaded_trainable_param_model.keys())
        overlap_names = set.intersection(loaded_trainable_param_names, frozen_param_names)
        assert len(overlap_names) == 0

        trainable_param_names = set([n for n, p in model.named_parameters() if p.requires_grad])
        assert loaded_trainable_param_names == trainable_param_names


class TestSaveTensorClone(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('zero_stage', [1, 2])
    @pytest.mark.parametrize('use_cpu_device', [True, False])
    def test_save_tensor_clone(self, tmpdir, zero_stage, use_cpu_device):

        ds_config = {
            "optimizer": {
                "type": "AdamW",
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1
        }
        dtype = torch.float16
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                dtype = torch.bfloat16
                ds_config["bf16"] = {"enabled": True}
            hpu_flag, msg = is_hpu_supported(ds_config)
            if not hpu_flag:
                pytest.skip(msg)
        hidden_dim = 1024
        model = SimpleModel(hidden_dim, nlayers=4).to(dtype=dtype)
        ref_model_state_dict = model.state_dict()

        ds_engine, _, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
        clone_device = torch.device('cpu') if use_cpu_device else get_accelerator().current_device()
        clone_state_dict = clone_tensors_for_torch_save(ds_engine.module.state_dict())
        compare_state_dicts(ref_model_state_dict, clone_state_dict)

        ref_ckpt_file = os.path.join(tmpdir, 'ref_ckpt.pt')
        torch.save(ref_model_state_dict, ref_ckpt_file)
        clone_ckpt_file = os.path.join(tmpdir, 'clone_ckpt.pt')
        torch.save(clone_state_dict, clone_ckpt_file)

        compare_state_dicts(torch.load(ref_ckpt_file), torch.load(clone_ckpt_file))


class TestZeRONonDistributed(DistributedTest):
    world_size = 1
    init_distributed = False

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    def test_chmod_exception_handling(self, monkeypatch, zero_stage):

        config_dict = {
            "optimizer": {
                "type": "AdamW"
            },
            "train_batch_size": 1,
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        args = SimpleNamespace(local_rank=0)
        net = SimpleModel(hidden_dim=4)
        engine, _, _, _ = deepspeed.initialize(args=args,
                                               config=config_dict,
                                               model=net,
                                               model_parameters=net.parameters())

        log_called = False

        def mock_logger_info(message, *args, **kwargs):
            nonlocal log_called
            log_called = True

        monkeypatch.setattr("deepspeed.utils.logger.info", mock_logger_info)
        """
            This is presented for use-cases like Azure Storage File Share (where permissions are not allowed)
            We use a fake file for this test (file not existing would present a similar issue as not being able to chmod)
        """
        fake_recovery_script_dst = os.path.join("tmp", "zero_to_fp32.py")
        engine._change_recovery_script_permissions(fake_recovery_script_dst)

        assert log_called, "Expected deepspeed.utils.logger.info to be called."
