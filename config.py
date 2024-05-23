config = {
    # Task
    "dimension": 4,
    "learn_function": "Sphere",
    "learn_function1": "Sphere",
    "learn_function2": "Abs",
    "test_function": "Abs",
    "budget": 2 * 4 + 2,
    # Functions number
    "batch": 1024,
    "num_batches": 20,
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
