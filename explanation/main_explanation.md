# Main Script Explanation (`main.py`)

This is the entry point of the training pipeline. It assembles the Configuration, Data, Model, and Training Loop.

## 1. Configuration Loading
```python
with open(os.path.join('config', dataset + '_config.yml'), "r") as ymlfile:
    c = yaml.load(ymlfile, Loader=yaml.FullLoader)
    conf = Struct(**c)
```
*   Loads hyperparameters (batch size, LR, Sigma, M, K) from a YAML file.
*   Converts dict to `Struct` for dot-notation access (`conf.lr`).

## 2. Environment Setup
*   **Device**: Selects CUDA if available.
*   **Reproducibility**: Sets manual seeds for `torch` and `numpy`.
    ```python
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    ```

## 3. Data Preparation
Instantiates the appropriate dataset based on the `dataset` variable.
```python
if dataset == 'mnist':
    train_data = MegapixelMNIST(conf, train=True)
    test_data = MegapixelMNIST(conf, train=False)
```
Creates PyTorch `DataLoader`s.
*   `persistent_workers=True`: Keeps worker processes alive to speed up epoch transitions.
*   `pin_memory=True`: Speeds up CPU-to-GPU transfer.

## 4. Model Initialization
```python
net = IDPSNet(conf).to(device)
```
Init the IDPS architecture with the loaded config.

## 5. Optimizer & Loss
*   **Optimizer**: `AdamW` (Adam with Weight Decay fix).
    *   Initial `lr=0` because the scheduler (`adjust_learning_rate`) sets it at the start of the loop.
*   **Criterions**:
    *   `NLLLoss`: For Multi-class classification (Softmax output).
    *   `BCELoss`: For Multi-label/Binary classification (Sigmoid output).

## 6. The Epoch Loop
Iterates `n_epoch` times.

```python
for epoch in range(conf.n_epoch):
    # Train
    train_one_epoch(...)
    log_writer_train.compute_metric()
    log_writer_train.print_stats(...)

    # Evaluate
    evaluate(...)
    log_writer_test.compute_metric()
    log_writer_test.print_stats(...)
```
Calls the training and evaluation functions defined in `iterative.py` and logs the results using the `Logger`.
