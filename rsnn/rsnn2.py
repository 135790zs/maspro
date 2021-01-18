import time
import numpy as np
from config2 import cfg as CFG
import utils2 as ut
import vis2 as vis



def main(cfg):
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

    print("N_train:", n_train, "N_val:", n_val)

    # Weights to the network. Keys: W, out, bias, B
    W = ut.initialize_weights(cfg=cfg, inp_size=n_channels, tar_size=n_phones)
    adamvars = ut.init_adam(cfg=cfg, tar_size=n_phones)
    betas = ut.initialize_betas(cfg=cfg)

    W_log = ut.initialize_W_log(cfg=cfg, W=W)

    for e in range(cfg["Epochs"]):

        # Train on batched input samples
        batch_idxs_list = ut.sample_mbatches(cfg=cfg, n_train=n_train)

        for b_idx, batch_idxs in enumerate(batch_idxs_list):
            print((f"{'-'*10} Epoch {e}/{cfg['Epochs']}\tBatch {b_idx}/"
                   f"{len(batch_idxs_list)} {'-'*10}"), end='\r')
            # Validate occasionally
            if adamvars['it'] % cfg["val_every_B"] == 0:
                valbatch = rng.choice(n_val, size=cfg["batch_size_val"])
                X = inps['val'][valbatch]
                T = tars['val'][valbatch]
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
            del M

            if (adamvars['it'] == 0
                or adamvars['it'] % cfg["plot_tracker_interval"] == 0):
                vis.plot_W(W_log=W_log,
                           cfg=cfg,
                           log_id=log_id)

            W, adamvars = ut.update_weights(cfg=cfg,
                                            e=e,
                                            W=W,
                                            G=G,
                                            adamvars=adamvars,
                                            it_per_e=len(batch_idxs_list))







if __name__ == "__main__":

    main(cfg=CFG)
