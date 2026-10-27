# Third-party notices

The EnvSched scheduler, training entry point, and scripts are released under the MIT license in `LICENSE`, which also keeps the RUC-NLPIR copyright notice for source derived from EnvScaler.

`vendor/ROLL` is a modified, reduced snapshot of Alibaba ROLL, including its mcore_adapter and the EnvScaler/BFCL integration. The Apache-2.0 license is preserved in `vendor/ROLL/LICENSE`, along with existing file-level notices. Local modifications support selector updates, EnvScaler trajectories, checkpoint handling, and the Qwen3 training runtime. Unused training pipelines, model families, and cluster-specific resource controls have been removed.

Training scenarios, executable environment metadata, and BFCL validation data are not included. `scripts/prepare_data.py` downloads them from EnvScaler (MIT) at a pinned commit; the BFCL data originates from the Berkeley Function-Calling Leaderboard and keeps its license. Task topologies are generated locally with `topology_generator/`. Existing notices on integrated third-party code remain applicable; this release does not relicense that code.

Qwen3 weights are separate downloads governed by their model licenses. Python dependencies, CUDA components, Apex, Transformer Engine, FlashAttention, and FlashInfer retain their respective licenses.
