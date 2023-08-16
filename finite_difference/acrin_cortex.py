import math

import numpy as np
import matplotlib.pyplot as plt


class ActinCortex:
    def __init__(self, I, L, T, dx, dt, params, user_action=None, completion_run=False, alpha_type="constant", validation=True):
        Nt = int(round(T/dt))
        Nx = int(round(L/dx))
        self.__dt = dt
        self.__dx = dx
        self._t = np.linspace(0, Nt*dt, Nt+1)      # Mesh points in time
        self._x = np.linspace(0, L, Nx+1)          # Mesh points in space

        self._params = params
        self.user_action = user_action
        self.completion_run = completion_run
        self.validation = validation

        self._p = np.vectorize(I[0], otypes=[float])(self._x)
        self._m = np.vectorize(I[1], otypes=[float])(self._x)
        self._cf = np.vectorize(I[2], otypes=[float])(self._x)
        self._cb = np.vectorize(I[3], otypes=[float])(self._x)
        self._v = np.zeros(np.size(Nx+1))

        alpha = params["alpha"]
        if alpha_type == "constant":
            self._alpha = alpha

        elif alpha_type == "tanh":
            alpha_func = lambda x: alpha * math.tanh(2/L * x)
            self._alpha = np.vectorize(alpha_func, otypes=[float])(self._x)

        else:
            raise ValueError("Bad input for alpha argument")
        
        if validation:
            self.monomers_sum = []
            self.cofilin_sum = []


    def get_f(self):
        """ Calculates the f part of the discretizied equations


        Returns:
            tuple: f_p[-1:1], f_m[-1:1], f_cf[-1:1], f_cb[-1:1]
        """
        dt, dx, p, cf, cb, v = self.__dt, self.__dx, self._p, self._cf, self._cb, self._v
        k_s = params["k_s"]
        alpha = self._alpha if isinstance(self._alpha, (int, float)) else self._alpha[1: -1]

        depoly = dt*alpha*cb[1:-1]*p[1:-1]
        bind = dt*k_s*cf[1:-1]*p[1:-1]

        f_p = - v*dt/(dx) * (p[1:-1] - p[:-2]) - depoly
        f_m = depoly
        f_cf = depoly - bind
        f_cb = - v*dt/dx * (cb[1: -1] - cb[:-2]) - f_cf
        
        return f_p, f_m, f_cf, f_cb


    def iterate_values(self,):
        
        dt, dx, p, m, cf, cb, v = self.__dt, self.__dx, self._p, self._m, self._cf, self._cb, self._v
        a_br, k_br, a_gr, k_gr, Dm, Dc, beta, m_c = self._params["a_br"], self._params["k_br"], self._params["a_gr"], \
              self._params["k_gr"], self._params["Dm"], self._params["Dc"], self._params["beta"], self._params["m_c"]

        v = a_br*k_br*(m[1]**2) + a_gr*k_gr*(m[1] - m_c)
        f = self.get_f()
        
        # Calculate boundaries
        p_L, m_L, cf_L, cb_L = 0, m[-2], cf[-2], cb[-2]
        p_0, cf_0, cb_0 = p[1], cf[1], 0
        m_0 = m[1] - (dx/(Dm*(1-beta*p[0]))) * p[1] * (a_br*k_br*m[1]**2 + a_gr*k_gr*(m[1]-m_c))

        # Calculate inner-points
        m_flux = Dm*dt/(2*dx**2) * \
                    ((2-beta*(p[1: -1] + p[2:]))*(m[2:] - m[1: -1]) - (2-beta*(p[1: -1] + p[:-2]))*(m[1: -1] - m[:-2]))
        cf_flux = Dc*dt/(dx**2) * (cf[2:] + cf[:-2] - 2*cf[1:-1])
        # Set boundaries
        p[-1], m[-1], cf[-1], cb[-1] = p_L, m_L, cf_L, cb_L
        p[0], m[0], cf[0], cb[0] = p_0, m_0, cf_0, cb_0

        # Assign new values
        p[1: -1] = p[1: -1] + f[0]
        m[1: -1] = m[1: -1] +  m_flux + f[1]
        cf[1: -1] = cf[1: -1] + cf_flux + f[2]
        cb[1: -1] = cb[1: -1] + f[3]

        return p, m, cf, cb, v
    

    def solver(self):
        """_summary_
    get_f - A function that returns a list of 4 values according to ${\bf f}$. The spatial derivatives would be calculated based on regular finite difference, e.g. $\frac{r_p^{n, m+1} - r_p^{n, m}}{\Delta x}$
        Args:

        Returns:
            _type_: _description_
        """
        dt, dx, p, m, cf, cb, v, t, x = self.__dt, self.__dx, self._p, self._m, self._cf, self._cb, self._v, self._t, self._x

        for n in t:
            if self.user_action:
                self.user_action(p=p, m=m, cf=cf, cb=cb, v=v, dx=dx, dt=dt, x=x, t=t, n=n)

            self.iterate_values()

        if self.completion_run:
            tolerance = params["tolerance"]

            completed = not self.completion_run
            while not completed:
                t = np.append(t, t[-1] + dt)
                if self.user_action:
                    self.user_action(p=p, m=m, cf=cf, cb=cb, v=v, dx=dx, dt=dt, x=x, t=t, n=t[-1])

                p_p, m_p, cf_p, cb_p, v_p = np.copy(p), np.copy(m), np.copy(cf), np.copy(cb), np.copy(v)
                self.iterate_values()

                checks = [
                    all(np.abs(p - p_p) < tolerance),
                    all(np.abs(m - m_p) < tolerance),
                    all(np.abs(cf - cf_p) < tolerance),
                    all(np.abs(cb - cb_p) < tolerance),
                ]
                completed = all(checks)

        return p, m, cf, cb, x, t
    

    def mass_sum(self):
        p, m, cf, cb, x = self._p, self._m, self._cf, self._cb, self._x

        sum_p = sum(p)/len(x)
        sum_m = sum(m)/len(x)
        sum_cb = sum(cb)/len(x)
        sum_cf = sum(cf)/len(x)

        self.monomers_sum.append(sum_m + sum_p)
        self.cofilin_sum.append(sum_cb + sum_cf)
        return 


if __name__ == "__main__":

    params = {
        "Dm": 8.4,
        "Dc": 10,
        "beta": 0.2,
        "k_s": 2,
        "k_gr": 8.7,
        "k_br": 2.16E-5,
        "a_gr": 2,
        "a_br": 2,
        "m_c": 0,
        "alpha": 2,
        "tolerance": 0.01
    }

    I = [
        lambda p: 0.2 * (p < 0.5),
        lambda m: 0.5,
        lambda cf: 0.15,
        lambda cb: 0 * (0 < cb < 0.5),
    ]

    T = 1
    L = 10
    dx = 0.1
    dt = 1e-5

    simulation = ActinCortex(I, L, T, dx, dt, params)
    sol = simulation.solver()
    print(sol)
