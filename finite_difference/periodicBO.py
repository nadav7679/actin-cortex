import numpy as np

from actin_cortex import ActinCortex


class ActinCortexPeriodicBO(ActinCortex):
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
        self._v = 1
        v = self._v
        f = self.get_f()

        m_next = np.roll(m, -2)
        m_back = np.roll(m, 2)
        p_next = np.roll(p, -2)
        p_back = np.roll(p, 2)
        cf_next = np.roll(cf, -2)
        cf_back = np.roll(cf, 2)

        m_flux = Dm*dt/(2*dx**2) * \
                    ((2-beta*(p + p_next))*(m_next - m) - (2-beta*(p + p_back))*(m - m_back))
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
        "beta": 0.8,
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
        lambda p: 0.5 * (1 < p < 3),
        lambda m: 0.2,
        lambda cf: 0.1,
        lambda cb: 0 * (0 < cb < 0.5),
    ]

    T = 20
    L = 10
    dx = 0.1
    dt = 1e-4

    simulation = ActinCortexPeriodicBO(I, L, T, dx, dt, params)
    sol = simulation.solver()
