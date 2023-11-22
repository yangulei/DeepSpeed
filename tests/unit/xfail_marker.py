# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

hpu_xfail_tests = {}

g1_xfail_tests = {
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[cross-encoder/ms-marco-MiniLM-L-12-v2-text-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/minilm-uncased-squad2-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/minilm-uncased-squad2-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[cross-encoder/ms-marco-MiniLM-L-12-v2-text-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/pythia-70m-deduped-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163095.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/pythia-70m-deduped-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163095.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/pythia-70m-deduped-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163095.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163099.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163099.",
    "unit/inference/test_inference.py::TestModelTask::test[distilgpt2-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[gpt2-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-fp32-CG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/gpt-j-6b-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[gpt2-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilgpt2-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-bf16-CG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/gpt-j-6b-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[1024-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[64-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[1048576-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[128-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[22-fp16]":
    "Xfail, due to SW-162575.",
    "unit/runtime/test_ds_initialize.py::TestOptimizerImplementation::test[fp16-fp32-zero3]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_ds_initialize.py::TestOptimizerImplementation::test[fp16-bf16-zero3]":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_model_quantization":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_post_init_quant":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_quantized_initialization_nvme_offload":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_post_init_quant_cpu_offload":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_half_int4_quantization":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_quantized_initialization_cpu_offload":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_quantized_linear":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_half_int8_quantization":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_quantized_initialization":
    "float16/half is not supported on Gaudi.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_post_init_quant_nvme_offload":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_pipeline_checkpoint_loading[3]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-3-dtype1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[none-3-dtype1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[True-MuAdamW-AdamW]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[True-MuSGD-SGD]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[False-MuAdam-Adam]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[False-MuAdamW-AdamW]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[True-MuAdam-Adam]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[False-MuSGD-SGD]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_save_exclude_frozen_weights[2]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_save_exclude_frozen_weights[1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-1-dtype1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-2-dtype1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[none-2-dtype1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[none-1-dtype1]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[2-20-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[1-8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[1-20-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[2-8-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[4-20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[4-8-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[4-20-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[1-8-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[2-8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[1-20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[2-20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TesthpZeroConfigSweep::test[4-8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[2-8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[2-20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[4-20-4000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[4-8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[2-8-4000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[4-20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[2-20-4000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_hpzero.py::TestSecondaryTensorSize::test[4-8-4000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qgzero.py::TesthpZeroConfigSweep::test[20-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qgzero.py::TesthpZeroConfigSweep::test[8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qgzero.py::TesthpZeroConfigSweep::test[20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qgzero.py::TesthpZeroConfigSweep::test[8-2000]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qwzero.py::TesthpZeroConfigSweep::test[8-2048]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qwzero.py::TesthpZeroConfigSweep::test[20-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qwzero.py::TesthpZeroConfigSweep::test[8-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_qwzero.py::TesthpZeroConfigSweep::test[20-2048]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[2]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[1]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[1]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[1]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[4]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[4]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[2]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[2]":
    "float16/half is not supported on Gaudi.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[4]":
    "float16/half is not supported on Gaudi.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[facebook/opt-350m]":
    "float16/half is not supported on Gaudi.",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[fp16-marian]":
    "float16/half is not supported on Gaudi.",
    "unit/inference/test_inference.py::TestLowCpuMemUsage::test[gpt2]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-9-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[2-4-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[1-9-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[2-9-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[1-4-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-4-1024]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-1232-255-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-4096-1-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-1232-1-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-1232-1-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-4096-255-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-4096-255-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-1232-255-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-1232-1-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-512-255-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-4096-128-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-1232-128-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-1232-128-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-512-1-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-1232-255-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-512-128-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-512-128-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-512-128-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-512-255-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-1232-128-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-1232-128-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-512-255-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-4096-128-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-512-1-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-4096-255-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-512-255-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-512-128-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-4096-1-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-4096-255-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-4096-1-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-1232-1-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-4096-128-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-512-1-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-4096-1-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[True-dtype0-512-1-1]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-4096-128-2]":
    "float16/half is not supported on Gaudi.",
    "unit/ops/transformer/inference/test_gelu.py::test_gelu[False-dtype0-1232-255-2]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-bigscience/bloom-560m-zero_stage=2-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-bigscience/bloom-560m-zero_stage=3-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-bigscience/bloom-560m-zero_stage=2-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-facebook/opt-350m-zero_stage=2-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-facebook/opt-350m-zero_stage=2-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-facebook/opt-350m-zero_stage=3-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-facebook/opt-350m-zero_stage=3-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-bigscience/bloom-560m-zero_stage=3-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-EleutherAI/gpt-neo-125m-zero_stage=2-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-EleutherAI/gpt-neo-125m-zero_stage=3-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-EleutherAI/gpt-neo-125m-zero_stage=3-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-EleutherAI/gpt-neo-125m-zero_stage=2-bsz=1]":
    "float16/half is not supported on Gaudi.",
    "unit/inference/test_inference.py::TestAutoTP::test[falcon]":
    "float16/half is not supported on Gaudi.",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[fp16-codegen]":
    "float16/half is not supported on Gaudi.",
    "unit/moe/test_moe.py::TestMoE::test[False-2-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-2-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-1-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-2-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-1-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-1-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-2-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-1-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-0-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[False-0-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[True-0-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[True-0-4]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_shared_weights.py::TestCheckpointSharedWeights::test_checkpoint_shared_weights":
    "Xfail, due to SW-162657.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config1]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config0]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config2]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_base[topo_config0]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_base[topo_config1]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_base[topo_config2]":
    "Xfail, due to SW-162653.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-bfp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-bfp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-fp32]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-fp32]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-bfp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-fp32]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroSupportedClientOptimizer::test[Adam]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroEmptyGrad::test":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[1-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[2-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[2-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[2-True]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[2-True]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[1-False]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestPRMoE::test[2-True]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestPRMoE::test[2-False]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-1-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-1-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-1-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-1-2]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[True-2]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[False-2]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[False]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[True]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_hybrid_optimizer_state[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[2-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[2-True-deepspeed_adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_hybrid_optimizer_state[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[2-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[2-True-deepspeed_adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_module_only[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[1-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[1-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_module_only[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe[4]":
    "Xfail, due to SW-162660.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-True]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-False]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-True]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-False]":
    "Xfail, due to SW-162650.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[True-3]":
    "Xfail, due to SW-148819.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[False-3]":
    "Xfail, due to SW-148819.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[3]":
    "Xfail, due to SW-148819.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[3]":
    "Xfail, due to SW-148819.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[3]":
    "Xfail, due to SW-148819.",
    "unit/runtime/zero/test_zero_context.py::TestSerialContext::test_subclass_param":
    "Xfail, due to SW-156783.",
    "unit/runtime/zero/test_zero_context_ancestry.py::TestSerialParamInit::test_subclass_param_init":
    "Xfail, due to SW-143227.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_fixed_dp[True-True-True]":
    "Xfail, due to SW-138014.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_fixed_dp[True-False-True]":
    "Xfail, due to SW-138014.",
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-neo]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-j]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[bf16-gpt-neo]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[bf16-gpt-j]":
    "Xfail, due to SW-162660.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[facebook/opt-1.3b-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[EleutherAI/gpt-neo-1.3B-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[EleutherAI/gpt-neo-1.3B-bsz=2]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[facebook/opt-1.3b-bsz=2]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=2]":
    "Xfail, due to SW-151621.",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-256-52-4-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-2048-128-32-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1600-128-25-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1600-128-25-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-128-128-2-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-True-True0]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-160-128-2-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-True-True1]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-2560-128-40-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-4096-128-64-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-160-128-2-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-120-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-512-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-1024-53-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1536-128-24-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-160-128-2-24-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-8192-128-64-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-511-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[1-256-2048-32-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[3-1024-54-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-1024-21-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1600-128-2-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[8-1600-128-2-3-True-True-0.05]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[8-160-128-2-3-True-True-0.1]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[8-1600-128-25-3-True-True-0.05]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[64-160-128-2-24-False-True-0.2]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[64-1600-128-2-4-False-True-0.2]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForwardSmallBatchSize::test_forward_with_small_bsz[8-7-1024-512-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForwardSmallBatchSize::test_forward_with_small_bsz[8-7-1024-512-16-3-False-True]":
    "CUDA tests not supported by HPU",
}

g2_xfail_tests = {
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[cross-encoder/ms-marco-MiniLM-L-12-v2-text-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/minilm-uncased-squad2-question-answering-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/minilm-uncased-squad2-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-cased-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/roberta-base-squad2-question-answering-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[cross-encoder/ms-marco-MiniLM-L-12-v2-text-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-uncased-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-base-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[dslim/bert-base-NER-token-classification-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[cross-encoder/ms-marco-MiniLM-L-12-v2-text-classification-fp16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[roberta-large-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-cased-fill-mask-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-uncased-fill-mask-fp32-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-fp16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-large-uncased-whole-word-masking-finetuned-squad-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[bert-base-multilingual-cased-fill-mask-bf16-CG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[deepset/minilm-uncased-squad2-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[Jean-Baptiste/roberta-large-ner-english-token-classification-fp32-noCG-noTriton]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/pythia-70m-deduped-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163095.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/pythia-70m-deduped-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163095.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/pythia-70m-deduped-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163095.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163099.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163099.",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163099.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-fp16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/gpt-j-6b-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilgpt2-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-fp32-CG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/gpt-j-6b-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[gpt2-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-bf16-CG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[EleutherAI/gpt-j-6b-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[gpt2-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[gpt2-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilgpt2-text-generation-fp32-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert-base-cased-distilled-squad-question-answering-fp16-CG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestModelTask::test[distilgpt2-text-generation-bf16-noCG-noTriton]":
    "Xfail, due to SW-163098.",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B]":
    "Xfail, due to SW-163102.",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-gpt2-xl]":
    "Xfail, due to SW-163104.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[facebook/opt-350m-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[facebook/opt-125m-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[bigscience/bloom-560m-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-neo-125M-int8]":
    "Xfail, due to SW-123615.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-j-6B-int8]":
    "Xfail, due to SW-123615.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-facebook/opt-350m-zero_stage=2-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-facebook/opt-350m-zero_stage=2-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[False-False]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[False-True]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[True-False]":
    "Xfail, due to SW-163097.",
    "unit/inference/test_model_profiling.py::TestModelProfiling::test[True-True]":
    "Xfail, due to SW-163097.",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-512-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1600-128-2-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-120-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-160-128-2-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-256-52-4-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-8192-128-64-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-1024-53-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-160-128-2-24-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-128-128-2-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1600-128-25-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-4096-128-64-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1536-128-24-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-True-True0]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-160-128-2-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-2048-128-32-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-2560-128-40-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1600-128-25-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[1-256-2048-32-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-1024-21-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-True-True1]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[3-1024-54-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-511-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[8-1600-128-2-3-True-True-0.05]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[64-1600-128-2-4-False-True-0.2]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[8-160-128-2-3-True-True-0.1]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[8-1600-128-25-3-True-True-0.05]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_backward.py::TestCUDABackward::test_backward[64-160-128-2-24-False-True-0.2]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForwardSmallBatchSize::test_forward_with_small_bsz[8-7-1024-512-16-3-False-True]":
    "CUDA tests not supported by HPU",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForwardSmallBatchSize::test_forward_with_small_bsz[8-7-1024-512-16-3-True-True]":
    "CUDA tests not supported by HPU",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[EleutherAI/gpt-neo-1.3B-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[facebook/opt-1.3b-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[EleutherAI/gpt-neo-1.3B-bsz=2]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_all.py::TestHybridEngineTextGen::test_functionality[facebook/opt-1.3b-bsz=2]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=1]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_llama.py::TestHybridEngineLlama::test_functionality[huggyllama/llama-7b-bsz=2]":
    "Xfail, due to SW-151621.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[cpu-EleutherAI/gpt-neo-125m-zero_stage=2-bsz=1]":
    "Xfail, due to SW-164229.",
    "unit/hybrid_engine/test_he_lora.py::TestHybridEngineLoRA::test_lora[none-EleutherAI/gpt-neo-125m-zero_stage=2-bsz=1]":
    "Xfail, due to SW-164229.",
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-j]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[bf16-gpt-neo]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[fp16-gpt-neo]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[bf16-gpt-j]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[fp16-bloom]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[fp16-gpt-j]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-neo]":
    "Xfail, due to SW-162660.",
    "unit/inference/test_inference.py::TestLowCpuMemUsage::test[gpt2]":
    "Xfail, due to SW-164236.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-4-1024]":
    "Xfail, due to SW-164239.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[2-4-1024]":
    "Xfail, due to SW-164239.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[1-9-1024]":
    "Xfail, due to SW-164239.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-9-1024]":
    "Xfail, due to SW-164239.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[1-4-1024]":
    "Xfail, due to SW-164239.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[2-9-1024]":
    "Xfail, due to SW-164239.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroEmptyGrad::test":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroSupportedClientOptimizer::test[Adam]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-fp32]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-fp32]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-bfp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-bfp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-fp32]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[default-fp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-fp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[bfp16-bfp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_bf16.py::TestZeroDtypeCocktail::test[fp16-fp16]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[True-2]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[True-3]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[True-1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-2]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-3]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-2]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-1-dtype1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-3-dtype1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-2-dtype1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[1]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[2]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-1]":
    "Xfail, due to SW-162657.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[1-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[2-True]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[2-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[1-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_lr_scheduler[2-False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_lr_scheduler.py::TestLRSchedulerCheckpoint::test_checkpoint_no_lr_scheduler[2-True]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestPRMoE::test[2-True]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestPRMoE::test[2-False]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[False-0-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[True-0-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[False-0-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe.py::TestMoE::test[True-0-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-1-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-1-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-2-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-True-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-1-2]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-True-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[True-False-1-4]":
    "Xfail, due to SW-162657.",
    "unit/moe/test_moe_tp.py::TestMOETensorParallel::test[False-False-1-2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_shared_weights.py::TestCheckpointSharedWeights::test_checkpoint_shared_weights":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[False-2]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[True-2]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[True]":
    "Xfail, due to SW-162657.",
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[False]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_not_load_optimizer_state[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_optimizer_state[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpointFrozenWeights::test_load_module_only[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_module_only[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[2-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_hybrid_optimizer_state[2]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[2-True-deepspeed_adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[2-True-deepspeed_adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_hybrid_optimizer_state[1]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_optimizer_state[1-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[1-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_not_load_optimizer_state[2-False-Adam]":
    "Xfail, due to SW-162657.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_load_module_only[1]":
    "Xfail, due to SW-162657.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[22-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[1048576-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[1024-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[128-fp16]":
    "Xfail, due to SW-162575.",
    "unit/ops/adam/test_cpu_adam.py::TestCPUAdam::test_fused_adam_equal[64-fp16]":
    "Xfail, due to SW-162575.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_unfused_gradnorm":
    "Xfail, due to SW-162660.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_post_init_quant_cpu_offload":
    "Xfail, due to SW-162660.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_quantized_initialization":
    "Xfail, due to SW-162660.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_post_init_quant":
    "Xfail, due to SW-162660.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_quantized_initialization_cpu_offload":
    "Xfail, due to SW-162660.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe[4]":
    "Xfail, due to SW-162660.",
    "unit/moe/test_moe.py::TestMoE::test[True-1-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-2-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-1-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-2-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-1-4]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-1-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[True-2-2]":
    "Xfail, due to SW-162650.",
    "unit/moe/test_moe.py::TestMoE::test[False-2-2]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-True]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-False]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-False]":
    "Xfail, due to SW-162650.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-True]":
    "Xfail, due to SW-162650.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config2]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config1]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config0]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_base[topo_config0]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_base[topo_config2]":
    "Xfail, due to SW-162653.",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_base[topo_config1]":
    "Xfail, due to SW-162653.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[False-3]":
    "Xfail, due to SW-100862.",
    "unit/runtime/zero/test_zero.py::TestZeroToFP32::test_2_param_groups[True-3]":
    "Xfail, due to SW-100862.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_immediate_save_load[3]":
    "Xfail, due to SW-100862.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_save_before_accum_grad_is_done[3]":
    "Xfail, due to SW-100862.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROSaveLoadEdgeCase::test_load_immediate_save[3]":
    "Xfail, due to SW-100862.",
    "unit/runtime/zero/test_zero_context.py::TestSerialContext::test_subclass_param":
    "Xfail, due to SW-156783.",
    "unit/runtime/zero/test_zero_context_ancestry.py::TestSerialParamInit::test_subclass_param_init":
    "Xfail, due to SW-143227.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_fixed_dp[True-True-True]":
    "Xfail, due to SW-138014.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_fixed_dp[True-False-True]":
    "Xfail, due to SW-138014.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-1-dtype1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-3-dtype1]":
    "Xfail, due to SW-145262.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[cpu-2-dtype1]":
    "Xfail, due to SW-145262.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_quantized_initialization_nvme_offload":
    "Xfail, due to SW-164545.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_zero3_int4_post_init_quant_nvme_offload":
    "Xfail, due to SW-164545.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[False-MuSGD-SGD]":
    "Xfail, due to SW-164551.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[True-MuSGD-SGD]":
    "Xfail, due to SW-164551.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[True-MuAdam-Adam]":
    "Xfail, due to SW-164551.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[False-MuAdam-Adam]":
    "Xfail, due to SW-164551.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[True-MuAdamW-AdamW]":
    "Xfail, due to SW-164551.",
    "unit/runtime/test_mup_optimizers.py::TestMuPOptimizers::test[False-MuAdamW-AdamW]":
    "Xfail, due to SW-164551.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[4]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[4]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[1]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[2]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[4]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[1]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_not_load_optimizer_state[2]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_optimizer_state[1]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_mics_optimizer.py::TestMiCSCheckpoint::test_load_module_only[2]":
    "Xfail, due to SW-164577.",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROCheckpoint::test_pipeline_checkpoint_loading[3]":
    "Xfail, due to SW-164593.",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[none-3-dtype1]":
    "Xfail, due to SW-164593.",
    "unit/inference/quantization/test_int4_quantization.py::TestQuantizedInt4::test_quantized_linear":
    "Xfail, due to SW-164606.",
}

gpu_xfail_tests = {
    "unit/moe/test_moe.py::TestMoE::test[False-2-2]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[False-1-2]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[True-2-2]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[True-1-4]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[True-2-4]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[True-1-2]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[False-1-4]": "Xfail, due to SW-163554.",
    "unit/moe/test_moe.py::TestMoE::test[False-2-4]": "Xfail, due to SW-163554.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-True]":
    "Xfail, due to SW-163554.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-True]":
    "Xfail, due to SW-163554.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-False]":
    "Xfail, due to SW-163554.",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-False]":
    "Xfail, due to SW-163554.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[bigscience/bloom-560m]":
    "Xfail, due to SW-163552.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[facebook/opt-125m]":
    "Xfail, due to SW-163552.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[facebook/opt-350m]":
    "Xfail, due to SW-163552.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-neo-125M]":
    "Xfail, due to SW-163552.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-j-6B]":
    "Xfail, due to SW-163552.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[1-4-1024]": "Xfail, due to SW-163551.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[1-9-1024]": "Xfail, due to SW-163551.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[2-4-1024]": "Xfail, due to SW-163551.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-4-1024]": "Xfail, due to SW-163551.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-9-1024]": "Xfail, due to SW-163551.",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[2-9-1024]": "Xfail, due to SW-163551.",
}
