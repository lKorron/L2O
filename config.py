config = {
    # Task
    "dimension": 4,
    "learn_function": "CustomComplexFunction",
    "test_function": "CustomComplexFunction",
    "budget": 2 * 4 + 2,
    "lower": -5,
    "upper": 5,
    # Functions number
    "batch": 1024,
    "num_batches": 3,
    "test_size": 100,
    # ML staff
    "train": True,
    "model": "CustomBatchedDropLSTM",
    "layers": 2,
    "lr": 3e-4,
    "hidden": 512,
    "epoch": 1,  # 5000
    "patience": 300,
    # Loss
    "last_impact": 2,
    "coef_scale": 5,
}
