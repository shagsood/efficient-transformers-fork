# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from QEfficient.generation._kv_dma_handoff import _prepare_kv_handoff_buffer


def test_short_prefill_kv_uses_zero_padded_decode_destination():
    destination = np.empty((1, 4, 4096, 132), dtype=np.float16)

    wire, copy_pair = _prepare_kv_handoff_buffer(
        destination.shape[:2] + (1024, 132), destination, np.dtype(np.float16)
    )

    assert wire.shape == (1, 4, 1024, 132)
    assert copy_pair[0] is wire
    assert copy_pair[1] is destination


def test_kv_handoff_rejects_incompatible_geometry():
    with pytest.raises(ValueError, match="KV buffer shape mismatch"):
        _prepare_kv_handoff_buffer(
            (1, 4, 1024, 132), np.empty((1, 8, 4096, 132), dtype=np.float16), np.dtype(np.float16)
        )
