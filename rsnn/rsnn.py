import time
import numpy as np
from config import cfg as CFG
# import utilsfast as ut
# import visfast as vis
import utils as ut
import vis as vis



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
    n_test = inps['test'].shape[0]

    print("N_train:", n_train, "N_val:", n_val, "N_test:", n_test)

    # Weights to the network. Keys: W, out, bias, B
    W = ut.initialize_weights(cfg=cfg, inp_size=n_channels, tar_size=n_phones)
    adamvars = ut.init_adam(cfg=cfg, tar_size=n_phones)
    betas = ut.initialize_betas(cfg=cfg)

    W_log = ut.initialize_W_log(cfg=cfg, W=W)

    best_val_ce = None
    flag_stop = False

    for e in range(cfg["Epochs"]):
        if flag_stop:
            break
        # Train on batched input samples
        batch_idxs_list = ut.sample_mbatches(cfg=cfg, n_samples=n_train, tvtype='train')

        for b_idx, batch_idxs in enumerate(batch_idxs_list):
            print((f"{'-'*10} Epoch {e}/{cfg['Epochs']}\tBatch {b_idx}/"
                   f"{len(batch_idxs_list)} {'-'*10}"))

            # Validate occasionally
            if adamvars['it'] % cfg["val_every_B"] == 0:
                Mv, avgs_v = feed_batch(
                    n_samples=n_val,
                    tvt_type='val',
                    cfg=cfg,
                    inps=inps['val'],
                    tars=tars['val'],
                    betas=betas,
                    W=W,
                    it=adamvars['it'],
                    best_val_ce=best_val_ce,
                    log_id=log_id)

                if (cfg["early_stopping"]
                        and best_val_ce is not None
                        and avgs_v['ce'] >= best_val_ce):
                    print(f"Stopping early, new ce {avgs_v['ce']:.3f} >= "
                          f"previous best {best_val_ce:.3f}")
                    flag_stop = True

                elif best_val_ce is None or avgs_v['ce'] < best_val_ce:
                    print(f"\nNew best validation error: {avgs_v['ce']:.3f} "
                          + ((f"<-- {best_val_ce:.3f}\n")
                             if best_val_ce is not None else '\n'))
                    best_val_ce = avgs_v['ce']
                    ut.save_checkpoint(W=W, cfg=cfg, log_id=log_id)

                best_val_ce = avgs_v['ce']
            else:
                avgs_v = None

            G, M, n_steps = ut.eprop(cfg=cfg,
                                     X=inps['train'][batch_idxs],
                                     T=tars['train'][batch_idxs],
                                     betas=betas,
                                     W=W)

            avgs_t = ut.get_avgs(M=M,
                               n_steps=n_steps)


            W_log = ut.update_W_log(W_log=W_log,
                                    W=W,
                                    log_id=log_id,
                                    avgs_t=avgs_t,
                                    avgs_v=avgs_v)

            if not cfg["visualize_val"] and (adamvars['it'] == 0
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
                vis.plot_GW(W=W,
                            G=G,
                            cfg=cfg,
                            log_id=log_id,
                            n_channels=n_channels,
                            n_phones=n_phones)

            if flag_stop:
                break

            W, adamvars = ut.update_weights(cfg=cfg,
                                            e=e,
                                            W=W,
                                            G=G,
                                            adamvars=adamvars,
                                            it_per_e=len(batch_idxs_list))


    # TEST

    Wopt = ut.load_checkpoint(log_id=log_id)

    _, avgs = feed_batch(
        tvt_type='test',
        n_samples=n_test,
        cfg=cfg,
        inps=inps['test'],
        tars=tars['test'],
        betas=betas,
        W=Wopt,
        best_val_ce=None,
        log_id=None,
        it=None)

    return avgs['ce'], 100-100*avgs['acc']


def feed_batch(tvt_type, n_samples, cfg, inps, tars, betas, W, best_val_ce, log_id, it):

    batch_idxs_list = ut.sample_mbatches(cfg=cfg, n_samples=n_samples, tvtype=tvt_type)

    for b_idx, batch_idxs in enumerate(batch_idxs_list):

        if b_idx == cfg[f"max_{tvt_type}_batches"]:
            break
        num_batches = min(len(batch_idxs_list), cfg[f"max_{tvt_type}_batches"])
        if num_batches < 0:
            num_batches = len(batch_idxs_list)

        print(f"{'-'*10} {tvt_type} batch {b_idx}/{num_batches} {'-'*10}")

        _, M, n_steps = ut.eprop(cfg=cfg,
                                 X=inps[batch_idxs],
                                 T=tars[batch_idxs],
                                 betas=betas,
                                 W=W,
                                 grad=False)
        avgs = ut.get_avgs(M=M,
                           n_steps=n_steps)

    return M, avgs

if __name__ == "__main__":

    ce, acc = main(cfg=CFG)
    print(f"Test cross-entropy: {ce:.3f}")
    print(f"Test error rate:    {acc:.1f}")
