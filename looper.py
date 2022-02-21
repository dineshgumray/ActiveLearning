# Active Learning Loop
num_queries = 10
for i in range(num_queries):
    # ...where each iteration consists of labelling 10 samples
    q_indices = active_learner.query(num_samples=10)

    # Simulate user interaction here. Replace this for real-world usage.
    y = train.y[q_indices]

    # Return the labels for the current query to the active learner.
    active_learner.update(y)

    labeled_indices = np.concatenate([q_indices, labeled_indices])

    print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
    results.append(evaluate(active_learner, train[labeled_indices], test))