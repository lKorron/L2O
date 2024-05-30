config = {
    # Task
    "dimension": 2,
    "learn_function": "Sphere",
    "test_function": "Sphere",
    "budget": 2 * 2 + 2,
    "lower": -50,
    "upper": 50,
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
