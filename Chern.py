import os
import multiprocessing as mp

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline


_pool_solver = None


def _init_pool_worker(solver):
    global _pool_solver
    _pool_solver = solver


def _pool_eval_t(t):
    s = _pool_solver
    u = s._chernoff_on_grid(t)
    return np.exp(-s.lam * t) * u


class ChernovSolver:
    """
    Параметры
    ---------
    a, b, c, g : callable
        Коэффициенты уравнения и правая часть (векторизованные).
    lam : float
        Параметр (lambda) в уравнении. 
    x_min, x_max : float
        Границы пространственной сетки усечения R.
    dx : float
        Шаг пространственной сетки.
    T : float, optional
        Верхний предел интегрирования по t. Если None, выбирается автоматически.
    n : int, optional
        Число шагов в композиции (S(t/n))^n.
    num_t : int, optional
        Число узлов временной сетки (должно быть нечётным для Симпсона).
    eps : float, optional
        Целевая точность усечения интеграла по t (используется в _auto_T).
    c_sup : float, optional
        Верхняя граница |c(x)|. 
    t_grading : str, optional
        Тип сетки по t: 'uniform' или 'quadratic' (сгущение у 0).
    interp : str, optional
        'cubic' (по умолчанию) или 'linear' — метод интерполяции u при
        применении S(τ).
    taper_width : float, optional
        Ширина приграничной полосы гладкого спада до нуля (используется
        только при interp='cubic'). По умолчанию 0.1*(x_max - x_min).
    """

    def __init__(self, a, b, c, g, lam, x_min, x_max, dx, T=None, n=100,
                 num_t=151, eps=1e-6, c_sup=None, t_grading='quadratic',
                 taper_width=None, interp='cubic'):
        self.a = a
        self.b = b
        self.c = c
        self.g = g
        self.lam = lam
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.T = T
        self.n = int(n)
        self.num_t = num_t
        self.eps = eps
        self.t_grading = t_grading
        self.taper_width = (0.1 * (x_max - x_min)
                             if taper_width is None else float(taper_width))
        if interp not in ('cubic', 'linear'):
            raise ValueError(f"Неизвестный interp: '{interp}'. "
                             "Используйте 'cubic' или 'linear'.")
        self.interp = interp

        # Пространственная сетка
        self.x_grid = np.arange(x_min, x_max + dx, dx)
        self.n_points = len(self.x_grid)

        # Кэширование коэффициентов
        self.a_vals = np.asarray(a(self.x_grid), dtype=float)
        self.b_vals = np.asarray(b(self.x_grid), dtype=float)
        self.c_vals = np.asarray(c(self.x_grid), dtype=float)
        self.g_vals = np.asarray(g(self.x_grid), dtype=float)

        # Оценка ||c||
        if c_sup is None:
            fine = np.linspace(x_min, x_max, 10 * self.n_points)
            self.c_sup = float(np.max(np.abs(c(fine))))
        else:
            self.c_sup = float(c_sup)

        # Проверка условия применимости
        if self.lam <= self.c_sup:
            raise ValueError(
                f"Условие λ > ‖c‖_∞ нарушено: λ = {self.lam}, "
                f"‖c‖_∞ ≈ {self.c_sup}. Интеграл может расходиться."
            )

        # Проверка эллиптичности
        a_min = float(np.min(self.a_vals))
        if a_min <= 0:
            raise ValueError(f"Условие a(x) > 0 нарушено: min a(x) = {a_min}.")
        self.a0 = a_min

    def _tapered_eval(self, cs, x_query):
        """
        Значения сплайна cs в точках x_query с гладким спадом до нуля вне [x_min, x_max].
        """
        vals = cs(x_query)
        mask = np.isnan(vals)
        if not np.any(mask):
            return vals

        x_lo = self.x_grid[0]
        x_hi = self.x_grid[-1]

        m = self.taper_width
        xo = x_query[mask]
        out = np.zeros_like(xo)

        left = xo < x_lo
        if np.any(left):
            d = x_lo - xo[left]
            edge_val = float(cs(x_lo))
            edge_der = float(cs(x_lo, 1))
            s = np.clip(1.0 - d / m, 0.0, 1.0)
            w = s * s * (3.0 - 2.0 * s)
            out[left] = (edge_val - edge_der * d) * w

        right = ~left
        if np.any(right):
            d = xo[right] - x_hi
            edge_val = float(cs(x_hi))
            edge_der = float(cs(x_hi, 1))
            s = np.clip(1.0 - d / m, 0.0, 1.0)
            w = s * s * (3.0 - 2.0 * s)
            out[right] = (edge_val + edge_der * d) * w

        vals[mask] = out
        return vals

    def _apply_S_on_grid(self, u_vals, tau):
        """
        Векторизованное применение оператора S(tau) к u, заданной в узлах сетки.
        """
        shift = 2.0 * np.sqrt(np.maximum(self.a_vals * tau, 0.0))
        x1 = self.x_grid + shift
        x2 = self.x_grid - shift
        x3 = self.x_grid + 2.0 * self.b_vals * tau

        if self.interp == 'cubic':
            cs = CubicSpline(self.x_grid, u_vals, extrapolate=False)
            u1 = self._tapered_eval(cs, x1)
            u2 = self._tapered_eval(cs, x2)
            u3 = self._tapered_eval(cs, x3)
        else:
            u1 = np.interp(x1, self.x_grid, u_vals, left=0.0, right=0.0)
            u2 = np.interp(x2, self.x_grid, u_vals, left=0.0, right=0.0)
            u3 = np.interp(x3, self.x_grid, u_vals, left=0.0, right=0.0)

        return 0.25 * u1 + 0.25 * u2 + 0.5 * u3 + tau * self.c_vals * u_vals

    def _chernoff_on_grid(self, t):
        """
        Возвращает (S(t/n))^n g на сетке x_grid.
        """
        if t <= 0.0:
            return self.g_vals.copy()
        tau = t / self.n
        u = self.g_vals.copy()
        for _ in range(self.n):
            u = self._apply_S_on_grid(u, tau)
        return u

    def _auto_T(self):
        """
        Автоматический выбор T 
        """
        delta = self.lam - self.c_sup
        T = np.log(2.0 / (delta * self.eps)) / delta
        return max(0.0, float(T))

    def _build_t_grid(self):
        """
        Сетка по t.
        """
        num_t = self.num_t if self.num_t % 2 == 1 else self.num_t + 1
        if self.t_grading == 'uniform':
            return np.linspace(0.0, self.T, num_t)
        elif self.t_grading == 'quadratic':
            s = np.linspace(0.0, 1.0, num_t)
            return self.T * s**2
        else:
            raise ValueError(f"Неизвестный t_grading: {self.t_grading}")

    def solve(self, n_jobs=1):
        """Вычисляет черновское приближение решения.

        Параметры
        ---------
        n_jobs : int, optional
            Число процессов. По умолчанию 1 (последовательно). n_jobs=-1 — все доступные ядра

        Returns
        -------
        f_chern : ndarray of shape (n_points,)
            Значения приближённого решения на пространственной сетке.
        """
        if self.T is None:
            self.T = self._auto_T()

        t_arr = self._build_t_grid()
        num_t = len(t_arr)

        # Подынтегральная функция
        if n_jobs == 1:
            F = np.empty((num_t, self.n_points))
            for i, t in enumerate(t_arr):
                u_t = self._chernoff_on_grid(t)
                F[i, :] = np.exp(-self.lam * t) * u_t
        else:
            workers = os.cpu_count() if n_jobs == -1 else int(n_jobs)
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers, initializer=_init_pool_worker,
                          initargs=(self,)) as pool:
                F = np.array(pool.map(_pool_eval_t, t_arr))

        # Интегрирование по t
        f_chern = simpson(F, x=t_arr, axis=0)
        return f_chern
