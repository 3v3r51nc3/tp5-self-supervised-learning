# TP5 – Self-supervised learning (M2 VMI, IFLCE015)

Comparison of self-supervised pretext tasks (Context Encoder, Rotation, SimCLR) and a new one,
**relative patch position** (Doersch et al., 2015), evaluated by linear probe on CIFAR10 and CIFAR100.
Built on top of [`ckurtz/cs294-158-ssl`](https://github.com/ckurtz/cs294-158-ssl) (`deepul_helper`).

## Results (linear probe, test top-1 %)

| Task | CIFAR10 | CIFAR100 (3 epochs) |
|---|---|---|
| Context Encoder | 45.77 | 13.09 |
| Rotation | 79.91 | 28.69 |
| SimCLR | 92.84 | 12.29 |
| Patch position (new) | 59.92 | 8.56 |

CIFAR10: Context Encoder, Rotation and SimCLR are the pretrained models given with the TP; patch position was trained here (100 epochs). CIFAR100 models only got 3 epochs each, so don't compare the two columns directly (see report).

## Layout

```
src/         patch_position.py (new task), apply_patches.py (plugs it + CIFAR100 into the lib), cifar100_hf.py (`prepare_cifar100`: gets CIFAR100 from Hugging Face, no slow official server)
notebooks/   tp5_cifar100_run.ipynb (Colab, full CIFAR100 run), demos_cifar10_run.ipynb (CIFAR10 demos)
results/     results_all.csv
scripts/     make_figure.py (csv -> report/figures/comparison.png)
report/      main.tex (French report) + figures/
docs/        assignment PDF and the Doersch et al. paper
```

## Reproduce

1. Open `notebooks/tp5_cifar100_run.ipynb` in Google Colab, GPU runtime (T4), Run all (~45 min).
   It clones the library, rebuilds CIFAR100 from Hugging Face, applies the patches, trains the 4 tasks
   (`EPOCHS` at the top) and prints the final table. Checkpoints are cached in Google Drive.
2. Update `results/results_all.csv`, then `python scripts/make_figure.py`.
3. Report: `cd report && latexmk -pdf main.tex` (put `Paris_Cité_Logo.png` next to `main.tex` for the logo).


