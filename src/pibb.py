def PIBB(Theta, Sigma, init_condit, tol=1e-3, max_iter=1000):
    # start_time = time.time()
    iter_count = 0
    delta_J = eval_rollout(task_info, Theta, init_condit)
    J_hist = [delta_J]

    Thetas = np.zeros(B * K * N).reshape((B, K, N))
    Ps = Js = np.zeros(K)

    while (iter_count < max_iter) and (abs(delta_J) > tol):
        iter_count += 1

        for k in range(K):

            Thetas[:, k, :] = np.array(
                [np.random.multivariate_normal(
                    Theta[:, n], Sigma[n, :, :]) for n in range(N)]
            ).transpose()

            Js[k] = eval_rollout(task_info, Thetas[:, k, :], init_condit)

        J_min = np.min(Js)
        J_max = np.max(Js)

        den = sum([np.exp(-h * (Js[l] - J_min) / (J_max - J_min))
                   for l in range(K)])
        for k in range(K):
            Ps[k] = np.exp(-h * (Js[k] - J_min) / (J_max - J_min)) / den

        Sigma = np.zeros(B * B * N).reshape((N, B, B))

        for n in range(N):
            for k in range(K):

                x = Ps[k] * \
                    np.matmul(
                        np.array([(Thetas[:, k, n] - Theta[:, n])]
                                 ).transpose(),
                        np.array([(Thetas[:, k, n] - Theta[:, n])])
                )
                Sigma[n, :, :] += x

            Sigma[n, :, :] = boundcovar(Sigma[n, :, :], lambda_min, lambda_max)

        Theta = np.zeros(B * N).reshape((B, N))

        for k in range(K):
            Theta += (Ps[k] * Thetas[:, k, :])

        J_hist.append(eval_rollout(task_info, Theta, init_condit))

        last_5_J = J_hist[-5:]  # is safe
        # function(works even when no of elements is less than 5)
        delta_J = np.mean(np.diff(last_5_J))

    return Theta, iter_count, J_hist