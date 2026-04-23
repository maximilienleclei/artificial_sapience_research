# STATUS.md

## Purpose

- Preserve the old MNIST classification idea as a tiny, rerunnable PyTorch example without porting the old Lightning/Hydra/W&B stack.

## Contents

- `code/train_mnist.py`: standalone PyTorch linear classifier over MNIST IDX gzip files.
- `data/raw/`: compressed MNIST files copied from `../ai_research/data/MNIST/raw`.
- `model/`: destination for trained checkpoint and metrics artifacts.

## Port Notes

- Source project: `../ai_research/projects/dl_classify_mnist`.
- Original setup used Lightning, Hydra configs, W&B media hooks, and `torchvision.datasets.MNIST`.
- Port keeps the same basic model shape from `task/fnn.yaml`: flattened 784-pixel input to 10 output classes.

## Verification

- Smoke test verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe`.
- Command: `python .\train_mnist.py --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256`
- Result: `test_accuracy=0.69043`, `final_train_loss=1.13252`, `examples_per_s=42273.7` on CPU.
- Artifacts: `model/mnist_linear.pt` and `model/metrics.json`.
- GPU smoke test verified on April 23, 2026 on the AMD Radeon RX 7800 XT machine using PyTorch `2.9.1+rocm7.2.1`; PyTorch exposes this ROCm GPU through `device=cuda`.
- GPU command: `python .\train_mnist.py --device cuda --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256 --metrics-path ..\model\metrics_cuda_smoke.json --model-path ..\model\mnist_linear_cuda_smoke.pt`
- GPU result: `test_accuracy=0.69043`, `final_train_loss=1.13252`, `examples_per_s=2609.9`. The tiny smoke test is slower on GPU because launch overhead dominates.
- The script now defaults to `--device auto`; on the AMD ROCm machine this resolves to `cuda`.

## Next Steps

- Run the full 3-epoch local baseline if this unit becomes relevant again.
