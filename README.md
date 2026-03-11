# draw_protein minimal project

This project is the smallest extracted subset I could isolate from `RFdiffusion` to support `scripts/draw_protein.py`.

Included:
- `scripts/draw_protein.py`
- `scripts/run_inference.py`
- `rfdiffusion/`
- `se3_transformer/`
- `sketch_process/` (trimmed from `sketchCNN.generate_mod`)
- `config/inference/curve.yaml`
- `models/Base_ckpt.pt`
- `models/Complex_base_ckpt.pt`
- `models/sketchcnn/ckpt/*`
- `examples/input_pdbs/1qys.pdb`
- `schedules/T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl`

Changes from the source repo:
- `draw_protein.py` now locates `run_inference.py` relative to the project root, so it can be called with an absolute path from any working directory.
- sketch generation no longer depends on the original absolute checkpoint path outside the repo.
- only the `process_curve()` runtime path was kept from `sketchCNN`; training and plotting code were removed.

Environment notes:
- The current shell `python` on this machine is Python 2.7. Use `python3` or your conda environment's Python when running this project.
- This extracted project was assembled statically and syntax-checked with `python3`, but not fully end-to-end runtime-tested in a prepared deep learning environment.
- To run inference you still need a Python environment with `torch`, `dgl`, `hydra-core`, `omegaconf`, `biopython`, `scipy`, `tensorflow`, and `tqdm`.
- Because `se3_transformer/` is vendored here, you do not need to separately install `env/SE3Transformer`.

Example:

```bash
cd /public/home/zhangyangroup/chengshiz/github/RFdiffusion_draw_protein_minimal
python3 scripts/draw_protein.py -c /path/to/curve.csv -n test_run
```

For `binder` and `drag`, also provide `-i /path/to/target.pdb`.
