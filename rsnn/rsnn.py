from config import cfg
import numpy as np
import utils as ut

# Variable arrays
Nv = np.ones(shape=(cfg["N_Rec"]+2, cfg["N_R"],)) * cfg["eqb"]
Nu = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))
Nz = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))
H = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))
TZ = np.zeros(shape=(cfg["N_Rec"]+2, cfg["N_R"],))

rng = np.random.default_rng()
W = rng.random(size=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))

EVv = np.zeros(shape=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))
EVu = np.zeros(shape=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))
ET = np.zeros(shape=(cfg["N_Rec"]+1, cfg["N_R"]*2, cfg["N_R"]*2,))


# R <- F(X, W_in)


# R <- G(R)


# Y = F(R, W_out)

for ep in range(0, 100):

    X = rng.random(size=(cfg["N_R"],))  # input is nonzero for first layer
    print("INPUT", X)

    for r in range(0, cfg["N_Rec"]-1):

        W[r, :, :] = ut.drop_weights(W=W[r, :, :],
                                     recur_lay1=(r > 0))
        Nvr, Nur, Nzr, EVv[r, :, :], EVu[r, :, :], Hr, W[r, :, :], \
            ET[r, :, :], TZr = ut.izh_eprop(
                Nv=np.concatenate((Nv[r, :], Nv[r+1, :])),
                Nu=np.concatenate((Nu[r, :], Nu[r+1, :])),
                Nz=np.concatenate((Nz[r, :], Nz[r+1, :])),
                TZ=np.concatenate((TZ[r, :], TZ[r+1, :])),
                H=np.concatenate((H[r, :], H[r+1, :])),
                EVv=EVv[r, :, :],
                EVu=EVu[r, :, :],
                ET=ET[r, :, :],
                W=W[r, :, :],
                X=np.pad(array=X, pad_width=(0, cfg["N_R"])),
                t=ep,
                rnd_factor=0)

        Nv[r, :] = Nvr[:cfg["N_R"]]
        Nu[r, :] = Nur[:cfg["N_R"]]
        Nz[r, :] = Nzr[:cfg["N_R"]]
        TZ[r, :] = TZr[:cfg["N_R"]]
        H[r, :] = Hr[:cfg["N_R"]]
        Nv[r+1, :] = Nvr[cfg["N_R"]:]
        Nu[r+1, :] = Nur[cfg["N_R"]:]
        Nz[r+1, :] = Nzr[cfg["N_R"]:]
        TZ[r+1, :] = TZr[cfg["N_R"]:]
        H[r+1, :] = Hr[cfg["N_R"]:]

        X = np.zeros(shape=(cfg["N_R"],))  # First layer passed, set input to 0

    print("OUTPUT", Nv[-1, :])
