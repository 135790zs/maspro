import time
import numpy as np
from config2 import cfg as CFG
import utils2 as ut
import vis2 as vis



def main(cfg):
    print(f"New thr: {cfg['thr']}")
    print(f"New alpha: {cfg['alpha']}")
    print(f"New beta: {cfg['beta']}")
    print(f"New rho: {cfg['rho']}")
    print(f"New kappaZ: {cfg['kappaZ']}")
    print(f"New kappaY: {cfg['kappaY']}")
    start_time = time.time()
    rng = np.random.default_rng(seed=cfg["seed"])

    log_id = ut.get_log_id()
    ut.prepare_log(cfg=cfg, log_id=log_id)

    # Load data
    inps, tars = ut.load_data(cfg=cfg)
    n_channels = inps['train'].shape[-1]
    n_phones = tars['train'].shape[-1]
    n_train = inps['train'].shape[0]
    n_val = inps['val'].shape[0]
    n_test = inps['test'].shape[0]

    print("N_train:", n_train, "N_val:", n_val, "N_test:", n_test)

    # Weights to the network. Keys: W, out, bias, B
    W = ut.initialize_weights(cfg=cfg, inp_size=n_channels, tar_size=n_phones)
    adamvars = ut.init_adam(cfg=cfg, tar_size=n_phones)
    betas = ut.initialize_betas(cfg=cfg)

    W_log = ut.initialize_W_log(cfg=cfg, W=W)

    best_val_ce = None

    for e in range(cfg["Epochs"]):

        # Train on batched input samples
        batch_idxs_list = ut.sample_mbatches(cfg=cfg, n_samples=n_train, tvtype='train')

        for b_idx, batch_idxs in enumerate(batch_idxs_list):
            print((f"{'-'*10} Epoch {e}/{cfg['Epochs']}\tBatch {b_idx}/"
                   f"{len(batch_idxs_list)} {'-'*10}"), end='\r')
            # Validate occasionally
            if adamvars['it'] % cfg["val_every_B"] == 0:
                valbatch = rng.choice(n_val, size=cfg["batch_size_val"])
                X = inps['val'][valbatch]
                T = tars['val'][valbatch]

                # TODO: Consider interpolating and trimming in eprop function itself.
                X, T = ut.interpolate_inputs(cfg=cfg,
                                             inp=X,
                                             tar=T,
                                             stretch=cfg["Repeats"])
                X, T = ut.trim_samples(X=X, T=T)

                n_steps = ut.count_lengths(X=X)

                _, Mv = ut.eprop(cfg=cfg,
                                 X=X,
                                 T=T,
                                 n_steps=n_steps,
                                 betas=betas,
                                 W=W)

                # Mean over batch and over time
                ce = np.mean(np.mean(Mv['ce'].cpu().numpy()/n_steps[:, None],
                                     axis=0),
                             axis=0)
                if best_val_ce is None or ce < best_val_ce:
                    print(f"\nBest new validation error: {ce:.3f} "
                          + (f"<-- {best_val_ce:.3f}\n")
                             if best_val_ce is not None else '\n')
                    best_val_ce = ce
                    ut.save_checkpoint(W=W, cfg=cfg, log_id=log_id)
            else:
                Mv = None

            X = inps['train'][batch_idxs]
            T = tars['train'][batch_idxs]

            X, T = ut.interpolate_inputs(cfg=cfg,
                                         inp=X,
                                         tar=T,
                                         stretch=cfg["Repeats"])

            X, T = ut.trim_samples(X=X, T=T)

            n_steps = ut.count_lengths(X=X)

            G, M = ut.eprop(cfg=cfg,
                            X=X,
                            T=T,
                            n_steps=n_steps,
                            betas=betas,
                            W=W)


            W_log = ut.update_W_log(W_log=W_log,
                                    Mt=M,
                                    Mv=Mv,
                                    W=W)

            if (adamvars['it'] == 0
                or adamvars['it'] % cfg["plot_model_interval"] == 0):
                vis.plot_M(M=M,
                           cfg=cfg,
                           it=adamvars['it'],
                           log_id=log_id,
                           n_steps=n_steps,
                           inp_size=n_channels)
            del M, Mv

            if (adamvars['it'] == 0
                or adamvars['it'] % cfg["plot_tracker_interval"] == 0):
                vis.plot_W(W_log=W_log,
                           cfg=cfg,
                           log_id=log_id)
                vis.plot_GW(W=W,
                            G=G,
                            cfg=cfg,
                            log_id=log_id,
                            n_channels=n_channels,
                            n_phones=n_phones)

            W, adamvars = ut.update_weights(cfg=cfg,
                                            e=e,
                                            W=W,
                                            G=G,
                                            adamvars=adamvars,
                                            it_per_e=len(batch_idxs_list))


    # TEST

    Wopt = ut.load_checkpoint(log_id=log_id)
    batch_idxs_list = ut.sample_mbatches(cfg=cfg, n_samples=n_test, tvtype='test')

    test_corr = 0
    test_ce = 0

    for b_idx, batch_idxs in enumerate(batch_idxs_list):
        print((f"{'-'*10} Test batch {b_idx}/"
               f"{len(batch_idxs_list)} {'-'*10}"), end='\r')
        # Validate occasionally

        X = inps['test'][batch_idxs]
        T = tars['test'][batch_idxs]

        X, T = ut.interpolate_inputs(cfg=cfg,
                                     inp=X,
                                     tar=T,
                                     stretch=cfg["Repeats"])

        X, T = ut.trim_samples(X=X, T=T)

        n_steps = ut.count_lengths(X=X)

        _, M = ut.eprop(cfg=cfg,
                        X=X,
                        T=T,
                        n_steps=n_steps,
                        betas=betas,
                        W=W)
        ce = np.mean(M['ce'].cpu().numpy(), axis=0)
        test_ce += np.mean(ce, axis=0) / len(batch_idxs_list)
        corr = np.mean(M['correct'].cpu().numpy(), axis=0)
        test_corr += np.mean(corr, axis=0) / len(batch_idxs_list)

    return test_ce, 1-test_corr





if __name__ == "__main__":

    ce, corr = main(cfg=CFG)
    print(f"Test cross-entropy: {ce:.3f}")
    print(f"Test error rate:    {100*corr:.1f}")
