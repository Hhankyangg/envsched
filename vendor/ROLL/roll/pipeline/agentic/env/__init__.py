"""Environment registration for EnvScaler training and BFCL validation."""
import gem

gem.register("envscaler_non_conv_env", entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerNonConvRLEnv")
gem.register("bfcl", entry_point="roll.pipeline.agentic.env.bfcl_env:BfclEnv")
