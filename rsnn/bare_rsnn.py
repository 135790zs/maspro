import numpy as np
from config import cfg as CFG
import utils as ut

def main(cfg):

    inps, tars = ut.load_data(cfg=cfg)
    n_channels = inps['train'].shape[-1]
    n_phones = tars['train'].shape[-1]
    n_train = inps['train'].shape[0]
    n_val = inps['val'].shape[0]

    rng = np.random.default_rng(seed=cfg["seed"])

    W = rng.normal(size=(cfg["N_R"], cfg["N_R"] + n_channels,)) / np.sqrt(cfg["N_R"] + n_channels)
    W_out = rng.normal(size=(n_phones, cfg["N_R"])) / np.sqrt(cfg["N_R"])
    bias = np.zeros(shape=(n_phones,))

    np.fill_diagonal(W[:, cfg["N_R"]:], 0)

    # Random B
    BA = rng.normal(size=(cfg["N_R"], n_phones,))

    # Adam
    mW = np.zeros_like(a=W)
    vW = np.zeros_like(a=W)
    mW_out = np.zeros_like(a=W_out)
    vW_out = np.zeros_like(a=W_out)
    mbias = np.zeros_like(a=bias)
    vbias = np.zeros_like(a=bias)

    # Initialize betas
    betas = np.zeros(shape=(cfg["N_R"]))
    betas[:int(betas.size * cfg["fraction_ALIF"])] = cfg["beta"]
    rng.shuffle(betas)

    for e in range(0, cfg["Epochs"]):

        mbsizes = [cfg["batch_size_train"]] * (n_train // cfg["batch_size_train"])
        mbsizes += [n_train % cfg["batch_size_train"]]

        for it, mbsize in enumerate(mbsizes):
            # IN BATCH
            idxs = np.arange(cfg["batch_size_train"]*it,
                             cfg["batch_size_train"]*it+mbsize)


            batch_inps = inps['train'][idxs]
            batch_tars = tars['train'][idxs]

            batch_inps, batch_tars = ut.interpolate_inputs(cfg=cfg,
                                                           inp=batch_inps,
                                                           tar=batch_tars,
                                                           stretch=cfg["Repeats"])
            # Crop -1's
            while np.all(batch_inps[:, -1] == -1):
                batch_inps = batch_inps[:, :-1, :]
            batch_tars = batch_tars[:, :batch_inps.shape[1]]

            n_steps = batch_inps.shape[1]

            # E-prop inits
            H = np.zeros(shape=(mbsize, cfg["N_R"],))
            V = np.zeros_like(a=H)
            a = np.zeros_like(a=H)
            A = np.zeros_like(a=H)  # Transient
            Z = np.zeros_like(a=H)
            Zs = np.zeros_like(a=H)
            Zbar = np.zeros_like(a=H)
            TZ = np.zeros_like(a=H)
            Y = np.zeros(shape=(mbsize, n_phones))  # transient
            D = np.zeros_like(a=Y)  # transient
            P = np.zeros_like(a=Y)  # transient
            CE = 0  # transient
            Z_in = np.zeros(shape=(mbsize, cfg["N_R"] + n_channels,))  # transient
            EVV = np.zeros(shape=(mbsize, cfg["N_R"], cfg["N_R"] + n_channels))
            EVA = np.zeros_like(a=EVV)
            ET = np.zeros_like(a=EVV)  # transient
            ETbar = np.zeros_like(a=EVV)
            gW = np.zeros_like(a=W)
            gW_out = np.zeros_like(a=W_out)
            gbias = np.zeros_like(a=bias)

            for t in range(n_steps):
                Z_in = np.concatenate((batch_inps[:, t], Z), axis=1)

                EVA = (np.einsum("bj, bji -> bji", H, EVV)
                       + (cfg["rho"] - np.einsum("bj, bji -> bji", H * betas, EVA)))
                EVV = cfg["alpha"] * np.einsum("bji, bi -> bji", EVV, Z_in)  # NOT SURE
                V = cfg["alpha"] * V + np.einsum("ji, bi -> bj", W, Z_in) - Z * cfg["thr"]
                a = cfg["rho"] * a + Z
                A = cfg["thr"] + betas * a
                Z = np.where(np.logical_and(t-TZ >= cfg["dt_refr"],
                                            V >= A),
                             1,
                             0)
                TZ[Z] = t
                Zs[Z] += 1
                H = 1 / cfg["thr"] * cfg["gamma"] * np.clip(
                    a=1-abs((V-A)/cfg["thr"]),
                    a_min=0,
                    a_max=None)
                ET = np.einsum("bj, bji -> bji", H, EVV) - np.einsum("j, bji -> bji", betas, EVA)
                ETbar = cfg["kappa"] * ETbar + ET
                Zbar = cfg["kappa"] * Zbar + Z
                Y = cfg["kappa"] * Y + np.einsum("kj, bj -> bk", W_out, Z) + bias
                P = np.exp(cfg["softmax_factor"] * (Y - np.max(Y)))
                P /= np.sum(P)
                CE = -np.sum(batch_tars[:, t] * np.log(1e-30 + P))
                D = P - batch_tars[:, t]
                L = np.einsum("jk, bk -> bj", BA, D)
                # n_steps below not precise, because differing lengths!
                L += cfg["FR_reg"] / n_steps * (Zs/(t+1) - cfg["FR_target"])
                gW += np.einsum("bj, bji -> ji", L, ETbar)
                gW_out += np.einsum("bk, bj -> kj", D, Zbar)

                gbias += np.sum(D)
            # Over time, caution: only count while not -1!
            print(gW.shape)
            gW = np.mean(gW)
            print(gW.shape)
            gW_out = np.mean(gW_out)
            gbias = np.mean(gbias)

            # Over batch
            gW = np.mean(gW)
            gW_out = np.mean(gW_out)
            gbias = np.mean(gbias)

            # Don't update dead weights
            gW[W==0] = 0

            # L2 regularization
            gW += cfg["L2_reg"] * np.linalg.norm(W.flatten())

            mW = cfg["adam_beta1"] * mW + (1 - cfg["adam_beta1"]) * gW
            vW = cfg["adam_beta2"] * vW + (1 - cfg["adam_beta2"]) * gW ** 2
            mWh = mW / (1 - cfg["adam_beta1"] ** (e+1))
            vWh = vW / (1 - cfg["adam_beta2"] ** (e+1))
            gW = gW - cfg["eta"] * (mWh / (np.sqrt(vWh) + cfg["adam_eps"]))

            mW_out = cfg["adam_beta1"] * mW_out + (1 - cfg["adam_beta1"]) * gW_out
            vW_out = cfg["adam_beta2"] * vW_out + (1 - cfg["adam_beta2"]) * gW_out ** 2
            mW_outh = mW_out / (1 - cfg["adam_beta1"] ** (e+1))
            vW_outh = vW_out / (1 - cfg["adam_beta2"] ** (e+1))
            gW_out = gW_out - cfg["eta"] * (mW_outh / (np.sqrt(vW_outh) + cfg["adam_eps"]))

            mbias = cfg["adam_beta1"] * mbias + (1 - cfg["adam_beta1"]) * gbias
            vbias = cfg["adam_beta2"] * vbias + (1 - cfg["adam_beta2"]) * gbias ** 2
            mbiash = mbias / (1 - cfg["adam_beta1"] ** (e+1))
            vbiash = vbias / (1 - cfg["adam_beta2"] ** (e+1))
            gbias = gbias - cfg["eta"] * (mbiash / (np.sqrt(vbiash) + cfg["adam_eps"]))


if __name__ == "__main__":

    main(cfg=CFG)
