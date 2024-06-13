config = {
    # Task
    "dimension": 4,
    "learn_function": "Sphere",
    "test_function": "Sphere",
    "budget": 2 * 4 + 2,
    "lower": -50,
    "upper": 50,
    # Functions number
    "batch": 3,
    "num_batches": 3,
    "test_size": 100,
    # ML staff
    "train": True,
    "model": "CleanLSTM",
    "layers": 2,
    "lr": 3e-4,
    "hidden": 512,
    "epoch": 5000,
    "patience": 300,
    # Loss
    "last_impact": 2,
    "coef_scale": 5,
}
