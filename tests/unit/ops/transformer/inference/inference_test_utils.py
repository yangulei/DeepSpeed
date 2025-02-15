# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from unit.hpu import *
from deepspeed.accelerator import get_accelerator

TOLERANCES = None


def get_tolerances():
    global TOLERANCES
    if TOLERANCES is None:
        TOLERANCES = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}
        if get_accelerator().is_bf16_supported():
            # Note: BF16 tolerance is higher than FP16 because of the lower precision (7 (+1) bits vs
            # 10 (+1) bits)
            TOLERANCES[torch.bfloat16] = (4.8e-1, 3.2e-2)
    return TOLERANCES


DTYPES = None


def get_dtypes():
    global DTYPES
    if DTYPES is None:
        DTYPES = [torch.float16, torch.float32]
        if bool(pytest.use_hpu) == True:
            if get_hpu_dev_version() == "Gaudi":
                DTYPES = [torch.float32]
        try:
            if get_accelerator().is_bf16_supported():
                DTYPES.append(torch.bfloat16)
        except (AssertionError, AttributeError):
            pass
    return DTYPES


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = get_tolerances()[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


def assert_almost_equal(x, y, decimal=2, err_msg=''):
    import numpy.testing as npt
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)
