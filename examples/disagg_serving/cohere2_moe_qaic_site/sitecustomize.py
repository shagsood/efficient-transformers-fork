"""Clear the QAIC default device group before child CLI parsing."""

try:
    import vllm_qaic.model_loader.qaic as loader

    for profile in ("default", "turbo"):
        loader.QAIC_DEVICE_CONFIG[profile]["device_group"] = None
except ImportError:
    pass
