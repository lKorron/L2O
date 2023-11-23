import operator


def test_black_box(model, black_box, rnn_iterations, start_point, start_hidden):
    timesteps = rnn_iterations
    x = start_point
    y = black_box(x)
    hidden = start_hidden

    results = []

    for _ in range(timesteps):
        x, y, hidden = model(black_box, x, y, hidden)
        results.append((x, y))
        print(f"x = {x.item()} y = {y.item()}")

    iteration, (best_x, best_y) = get_best_iteration(results, black_box)

    print(f"x = {best_x.item()} y = {best_y.item()} at iteration {iteration}")

    return iteration


def get_best_iteration(results, function, epsilon=0.1):
    max_tuple = min(results, key=operator.itemgetter(1))
    iteration = results.index(max_tuple) + 1
    # max_tuple = results[9]
    # iteration = 10
    #
    # for i, (x, y) in enumerate(results):
    #     if abs(y - function(0)) <= epsilon:
    #         iteration = i + 1
    #         max_tuple = (x, y)
    #         break

    return iteration, max_tuple


def get_norm(model):
    total_norm = 0

    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm
