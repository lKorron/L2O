config = {
    # Task
    "dimension": 2,
    "learn_function": "Rastrigin",
    "test_function": "Rastrigin",
    "budget": 2 * 3 + 1,
    "lower": -5,
    "upper": 5,
    # Functions number
    "batch": 1024,
    "num_batches": 3,
    "test_size": 100,
    # ML staff
    "train": True,
    "model": "CustomLSTM",
    "layers": 2,
    "lr": 3e-4,
    "hidden": 512,
    "epoch": 5000,
    "patience": 300,
    # Loss
    "last_impact": 2,
    "coef_scale": 5,
}
