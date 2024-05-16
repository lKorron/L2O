config = {
    # Task
    "dimension": 2,
    "learn_function": "Rosenbrock",
    "test_function": "Rosenbrock",
    "budget": 2 * 3 + 1,
    # Functions number
    "batch": 1024,
    "num_batches": 3,
    "test_size": 100,
    # ML staff
    "train": False,
    "model": "CustomLSTM",
    "layers": 2,
    "lr": 3e-4,
    "hidden": 512,
    "epoch": 5000,
    "patience": 300,
}
