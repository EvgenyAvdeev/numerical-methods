import numpy as np
from scipy.linalg import solve_banded
from scipy.linalg import solve as dense_solve


def _scalar(val):
    """
    Приводит любой результат callable(scalar_x) к float.
    """
    return float(np.asarray(val).flat[0])


def _array(val):
    """
    Приводит любой результат callable(array_x) к ndarray float.
    """
    return np.asarray(val, dtype=float)


class FiniteDifferenceSolver:
    """
    Параметры
    ----------
    a, b, c, g : callable
        Коэффициенты и правая часть.
    lam : float
        Параметр lamda.
    x_min, x_max : float
        Границы отрезка.
    dx : float
        Шаг сетки.
    bc_left : tuple, optional
        ГУ на левой границе. ('dirichlet', val) или ('neumann', val).
        По умолчанию ('dirichlet', 0.0).
    bc_right : tuple, optional
        ГУ на правой границе. По умолчанию ('dirichlet', 0.0).

    Примеры
    --------
    >>> solver = FiniteDifferenceSolver(a, b, c, g, lam, 0, 1, 0.01,
    ...     bc_left=('dirichlet', 0.0),
    ...     bc_right=('neumann', 0.0))
    >>> f = solver.solve()
    """

    def __init__(self, a, b, c, g, lam, x_min, x_max, dx,
                 bc_left=('dirichlet', 0.0),
                 bc_right=('dirichlet', 0.0)):
        self.a = a
        self.b = b
        self.c = c
        self.g = g
        self.lam = lam
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.bc_left  = bc_left
        self.bc_right = bc_right

        self.x_grid  = np.arange(x_min, x_max + dx / 2, dx)
        self.n_points = len(self.x_grid)

        self.a_vals = _array(a(self.x_grid))
        self.b_vals = _array(b(self.x_grid))
        self.c_vals = _array(c(self.x_grid))
        self.g_vals = _array(g(self.x_grid))

    def _build_system(self):
        N   = self.n_points
        h   = self.dx
        h2  = h * h

        a_v = self.a_vals
        b_v = self.b_vals
        c_v = self.c_vals
        g_v = self.g_vals

        alpha = a_v / h2 - b_v / (2 * h)         # при f_{i-1}
        beta  = -2 * a_v / h2 + (c_v - self.lam) # диагональ
        gamma = a_v / h2 + b_v / (2 * h)         # при f_{i+1}
        rhs   = -g_v.copy()

        ab = np.zeros((3, N))
        ab[1, :]   = beta
        ab[0, 1:]  = gamma[:-1]
        ab[2, :-1] = alpha[1:]

        bc = self.bc_left
        kind = bc[0]
        if kind == 'dirichlet':
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
            rhs[0]   = bc[1]
        elif kind == 'neumann':
            ab[0, 1] += alpha[0]
            rhs[0]   += 2.0 * h * alpha[0] * bc[1]
        else:
            raise ValueError(f"Неизвестный тип ГУ: '{kind}'. "
                             "Используйте 'dirichlet' или 'neumann'.")

        bc = self.bc_right
        kind = bc[0]
        if kind == 'dirichlet':
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
            rhs[-1]   = bc[1]
        elif kind == 'neumann':
            ab[2, -2] += gamma[-1]
            rhs[-1]   -= 2.0 * h * gamma[-1] * bc[1]
        else:
            raise ValueError(f"Неизвестный тип ГУ: '{kind}'. "
                             "Используйте 'dirichlet' или 'neumann'.")

        return ab, rhs

    def solve(self):
        ab, rhs = self._build_system()
        return solve_banded((1, 1), ab, rhs)


class _ShootingBase:
    """
    Общая логика метода стрельбы для RK4 и Розенброка.

    Поддерживаемые режимы (определяются параметрами bc_left / bc_right):

    1. Задача Коши (оба условия слева):
         ic = (f0, fp0)  →  f(x_min)=f0, f'(x_min)=fp0
       Стрельба не нужна — прямое интегрирование.

    2. Двухточечная краевая задача (по одному ГУ с каждой стороны):
         bc_left  = ('dirichlet'|'neumann', val) — условие на x_min
         bc_right = ('dirichlet'|'neumann', val) — условие на x_max

       Линейная стрельба, два прогона интегратора:

         Y_part : неоднородное решение (g≠0) с y0, в котором известная
                  компонента левого ГУ равна заданному значению, а свободная
                  равна 0;
         Y_hom  : однородное решение (g=0) с y0, дающим единицу по свободной
                  компоненте и ноль по известной (это масштабируемая на s
                  часть решения).

       Параметр стрельбы:
              s = − R_part / R_hom_lin,
       где R_part — невязка правого ГУ для Y_part (с учётом свободного члена
       правого ГУ), R_hom_lin — линейная часть оператора правого ГУ, применённая
       к Y_hom. Итог:  f(x) = Y_part(x) + s · Y_hom(x).

    Параметры
    ----------
    a, b, c, g : callable
    lam        : float
    x_min, x_max, dx : float
    ic         : tuple (f0, fp0) или None
        Начальные условия задачи Коши. Если задан, bc_left/bc_right игнорируются.
    bc_left    : tuple
        Граничное условие на x_min (для краевой задачи).
    bc_right   : tuple
        Граничное условие на x_max (для краевой задачи).
    """

    def __init__(self, a, b, c, g, lam, x_min, x_max, dx,
                 ic=None,
                 bc_left=('dirichlet', 0.0),
                 bc_right=('dirichlet', 0.0)):
        self.a = a
        self.b = b
        self.c = c
        self.g = g
        self.lam    = lam
        self.x_min  = x_min
        self.x_max  = x_max
        self.dx     = dx
        self.ic     = ic
        self.bc_left  = bc_left
        self.bc_right = bc_right

        self.x_grid   = np.arange(x_min, x_max + dx / 2, dx)
        self.n_points = len(self.x_grid)

    def _integrate(self, y0, use_g: bool) -> np.ndarray:
        raise NotImplementedError

    def _y0_part(self, bc):
        kind = bc[0]
        if kind == 'dirichlet':
            return np.array([bc[1], 0.0])
        elif kind == 'neumann':
            return np.array([0.0, bc[1]])
        else:
            raise ValueError(f"Неизвестный тип левого ГУ: '{kind}'. "
                             "Поддерживаются 'dirichlet' и 'neumann'.")

    def _y0_hom(self, bc):
        kind = bc[0]
        if kind == 'dirichlet':
            return np.array([0.0, 1.0])
        elif kind == 'neumann':
            return np.array([1.0, 0.0])
        else:
            raise ValueError(f"Неизвестный тип левого ГУ: '{kind}'.")

    def _bc_residual(self, Y, bc):
        kind = bc[0]
        f_val  = Y[-1, 0]
        fp_val = Y[-1, 1]
        if kind == 'dirichlet':
            return f_val - bc[1]
        elif kind == 'neumann':
            return fp_val - bc[1]
        else:
            raise ValueError(f"Неизвестный тип правого ГУ: '{kind}'.")

    def _bc_linear(self, Y, bc):
        kind = bc[0]
        f_val  = Y[-1, 0]
        fp_val = Y[-1, 1]
        if kind == 'dirichlet':
            return f_val
        elif kind == 'neumann':
            return fp_val
        else:
            raise ValueError(f"Неизвестный тип правого ГУ: '{kind}'.")

    def solve(self):
        if self.ic is not None:
            f0, fp0 = float(self.ic[0]), float(self.ic[1])
            Y = self._integrate(np.array([f0, fp0]), use_g=True)
            return Y[:, 0]

        Y_part = self._integrate(self._y0_part(self.bc_left), use_g=True)
        Y_hom  = self._integrate(self._y0_hom (self.bc_left), use_g=False)

        R_part = self._bc_residual(Y_part, self.bc_right)
        R_hom  = self._bc_linear  (Y_hom,  self.bc_right)

        if abs(R_hom) < 1e-14:
            raise RuntimeError(
                "Однородная часть решения не удовлетворяет правому ГУ "
                "ни при каком s. Задача вырождена или λ некорректен."
            )

        s = -R_part / R_hom
        return Y_part[:, 0] + s * Y_hom[:, 0]


class RungeKuttaSolver(_ShootingBase):
    """
    Параметры
    ----------
    a, b, c, g : callable
    lam        : float
    x_min, x_max, dx : float
    ic         : tuple (f0, fp0), optional
        Задача Коши: f(x_min)=f0, f'(x_min)=fp0.
        Если задан, bc_left/bc_right игнорируются.
    bc_left    : tuple, optional
        ГУ на левой границе (для краевой задачи).
        ('dirichlet', val) или ('neumann', val). По умолчанию ('dirichlet', 0.0).
    bc_right   : tuple, optional
        ГУ на правой границе. По умолчанию ('dirichlet', 0.0).

    Примеры
    --------
    # Краевая задача: f(0)=1, f(1)=0
    >>> solver = RungeKuttaSolver(a, b, c, g, lam, 0, 1, 0.01,
    ...     bc_left=('dirichlet', 1.0), bc_right=('dirichlet', 0.0))

    # Задача Коши: f(0)=1, f'(0)=0
    >>> solver = RungeKuttaSolver(a, b, c, g, lam, 0, 1, 0.01,
    ...     ic=(1.0, 0.0))
    """

    def _rhs(self, x, y, use_g):
        ax = _scalar(self.a(x))
        bx = _scalar(self.b(x))
        cx = _scalar(self.c(x))
        gx = _scalar(self.g(x)) if use_g else 0.0
        fpp = (-bx * y[1] - (cx - self.lam) * y[0] - gx) / ax
        return np.array([y[1], fpp])

    def _integrate(self, y0, use_g):
        N = self.n_points
        h = self.dx
        Y = np.zeros((N, 2))
        Y[0] = y0
        for i in range(N - 1):
            xi = self.x_grid[i]
            yi = Y[i]
            k1 = self._rhs(xi,         yi,             use_g)
            k2 = self._rhs(xi + h/2,   yi + h/2 * k1,  use_g)
            k3 = self._rhs(xi + h/2,   yi + h/2 * k2,  use_g)
            k4 = self._rhs(xi + h,     yi + h   * k3,  use_g)
            Y[i + 1] = yi + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        return Y

class RosenbrockSolver(_ShootingBase):
    """
    Параметры
    ----------
    a, b, c, g : callable
    lam        : float
    x_min, x_max, dx : float
    ic         : tuple (f0, fp0), optional
        Задача Коши: f(x_min)=f0, f'(x_min)=fp0.
    bc_left    : tuple, optional
        ГУ на левой границе. По умолчанию ('dirichlet', 0.0).
    bc_right   : tuple, optional
        ГУ на правой границе. По умолчанию ('dirichlet', 0.0).

    Примеры
    --------
    # Смешанные ГУ: f(0)=2, f'(1)=0
    >>> solver = RosenbrockSolver(a, b, c, g, lam, 0, 1, 0.005,
    ...     bc_left=('dirichlet', 2.0), bc_right=('neumann', 0.0))

    # Задача Коши
    >>> solver = RosenbrockSolver(a, b, c, g, lam, 0, 1, 0.005,
    ...     ic=(0.5, -1.0))
    """

    _GAMMA = 1.0 - 1.0 / np.sqrt(2.0)  

    def _rhs(self, x, y, use_g):
        ax = _scalar(self.a(x))
        bx = _scalar(self.b(x))
        cx = _scalar(self.c(x))
        gx = _scalar(self.g(x)) if use_g else 0.0
        fpp = (-bx * y[1] - (cx - self.lam) * y[0] - gx) / ax
        return np.array([y[1], fpp])

    def _jacobian(self, x):
        ax = _scalar(self.a(x))
        bx = _scalar(self.b(x))
        cx = _scalar(self.c(x))
        return np.array([
            [0.0,                   1.0      ],
            [-(cx - self.lam) / ax, -bx / ax ]
        ])

    def _integrate(self, y0, use_g):
        N   = self.n_points
        h   = self.dx
        gam = self._GAMMA
        Y   = np.zeros((N, 2))
        Y[0] = y0.copy()

        for i in range(N - 1):
            xi = self.x_grid[i]
            yi = Y[i]

            J = self._jacobian(xi)
            M = np.eye(2) - gam * h * J   # (I - γhJ)

            k1 = dense_solve(M, self._rhs(xi, yi, use_g))
            k2 = dense_solve(M, self._rhs(xi + h, yi + h * k1, use_g) - 2 * k1)

            Y[i + 1] = yi + h * (1.5 * k1 + 0.5 * k2)

        return Y
