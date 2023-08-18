import numpy as np

from actin_cortex import ActinCortex


class ActinCortexPeriodicBO(ActinCortex):
    def get_f(self):
        """Calculates the f part of the discretizied equations

        Returns:
            tuple: f_p[-1:1], f_m[-1:1], f_cf[-1:1], f_cb[-1:1]
        """
        dt, dx, p, cf, cb, v = (
            self._dt,
            self._dx,
            self._p,
            self._cf,
            self._cb,
            self._v,
        )
        k_s = self._params["k_s"]
        alpha = (
            self._alpha if isinstance(self._alpha, (int, float)) else self._alpha[1:-1]
        )

        depoly = (dt * alpha) * cb * p
        bind = (dt * k_s) * cf * p

        # p_prev = np.roll(p, -1)
        p_next = np.roll(p, 1)
        # cb_prev = np.roll(cb, -1)
        cb_next = np.roll(cb, 1)

        f_p = -v * dt / (dx) * (p_next - p) - depoly  # From some reason second ordergives an
        f_m = depoly                                  # unstavle solution. Also v has to be positive
        f_cf = depoly - bind
        f_cb = -v * dt / (dx) * (cb_next - cb) - f_cf

        return f_p, f_m, f_cf, f_cb

    def iterate_values(self):
        dt, dx, p, m, cf, cb, v = (
            self._dt,
            self._dx,
            self._p,
            self._m,
            self._cf,
            self._cb,
            self._v,
        )

        a_br, k_br, a_gr, k_gr, Dm, Dc, beta, m_c = (
            self._params["a_br"],
            self._params["k_br"],
            self._params["a_gr"],
            self._params["k_gr"],
            self._params["Dm"],
            self._params["Dc"],
            self._params["beta"],
            self._params["m_c"],
        )
        self._v = -1
        v = self._v
        f = self.get_f()

        m_next = np.roll(m, 2)
        m_back = np.roll(m, -2)
        m_flux = Dm * dt / (dx**2) * (m_next + m_back - 2 * m)

        cf_next = np.roll(cf, 2)
        cf_back = np.roll(cf, -2)
        cf_flux = Dc * dt / (dx**2) * (cf_back + cf_next - 2 * cf)

        # Assign new values
        p += f[0]
        m += m_flux + f[1]
        cf += cf_flux + f[2]
        cb += f[3]

        # Calculate boundaries
        p_L, m_L, cf_L, cb_L = p[0], m[0], cf[0], cb[0]
        p_0, m_0, cf_0, cb_0 = p[-1], m[-1], cf[-1], cb[-1]

        # Set boundaries
        p[-1], m[-1], cf[-1], cb[-1] = p_L, m_L, cf_L, cb_L
        p[0], m[0], cf[0], cb[0] = p_0, m_0, cf_0, cb_0

        return p, m, cf, cb, v


if __name__ == "__main__":
    params = {
        "Dm": 8.4,
        "Dc": 10,
        "beta": 0.2,
        "k_s": 2,
        "k_gr": 8.7,
        "k_br": 2.16e-5,
        "a_gr": 2,
        "a_br": 2,
        "m_c": 0,
        "alpha": 2,
        "tolerance": 0.01,
    }

    I = [
        lambda p: 0.6 * (1 < p < 3),
        lambda m: 0,
        lambda cf: 0.5,
        lambda cb: 0 * (0 < cb < 0.5),
    ]

    T = 5
    L = 10
    dx = 0.1
    dt = 1e-5

    simulation = ActinCortexPeriodicBO(I, L, T, dx, dt, params)
    sol = simulation.solver()
