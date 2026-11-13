import numpy as np
import sympy as sp

X_MIN, X_MAX = -1.0, 1.0

x = sp.symbols('x', real=True)


def _to_vectorized(expr):
    """sympy expr -> callable(x_array) -> ndarray той же формы, что и x_array.

    Обычный sp.lambdify для констант (например a(x) = 1) возвращает скаляр
    независимо от формы входного массива — это ломает код, ожидающий массив
    (например индексацию alpha[0] в FiniteDifferenceSolver). Здесь результат
    всегда явно приводится к форме входа.
    """
    f = sp.lambdify(x, expr, modules=['numpy'])

    def wrapped(xv):
        xv = np.asarray(xv, dtype=float)
        val = np.asarray(f(xv), dtype=float)
        if val.shape != xv.shape:
            val = np.broadcast_to(val, xv.shape).copy()
        return val

    return wrapped


def glued(f1_expr, tail_expr, match_order=2, alpha=1.0):
    """Piecewise-решение: f1_expr при x<=0, а при x>=0 — (P(x) +
    x**(match_order+1)*tail_expr) * exp(-alpha*x**2), где полином P(x)
    степени match_order подобран так, что ряд Тейлора P(x)*exp(-alpha x^2)
    в нуле совпадает с рядом Тейлора f1_expr до порядка match_order
    включительно.

    Важно: правая ветвь ОБЯЗАНА быть глобально ограниченной на R (условие
    f in UC_b(R) в task.md) — это исключает "голое" полиномиальное
    продолжение ряда Тейлора (f2 = Taylor(x) + x^(k+1)*tail): такой полином
    неограниченно растёт при x -> +oo. Внутри номинального окна [-1,1] это
    может быть незаметно, но ChernovSolver считает на расширенном окне
    (с запасом под черновский сдвиг), и там расхождение сразу проявляется.
    Множитель exp(-alpha x^2) даёт глобальное затухание, при этом не мешая
    подбору P под совпадение производных в нуле (решается символьно).

    Слагаемое x**(match_order+1)*tail_expr и все его производные до порядка
    match_order включительно равны нулю в x=0, поэтому склейка автоматически
    непрерывна вместе с производными до match_order. Производная порядка
    match_order+1 в общем случае терпит разрыв в x=0 — это управляемый
    параметр "гладкости стыка".
    """
    targets = [sp.diff(f1_expr, x, k).subs(x, 0) / sp.factorial(k)
               for k in range(match_order + 1)]
    p_coeffs = sp.symbols(f'p0:{match_order + 1}')
    P = sum(p_coeffs[i] * x**i for i in range(match_order + 1))
    envelope = sp.exp(-alpha * x**2)
    series = [sp.diff(P * envelope, x, k).subs(x, 0) / sp.factorial(k)
              for k in range(match_order + 1)]
    sol = sp.solve([sp.Eq(series[k], targets[k]) for k in range(match_order + 1)],
                    list(p_coeffs))
    P_solved = P.subs(sol)
    f2_expr = sp.simplify((P_solved + x**(match_order + 1) * tail_expr) * envelope)
    return sp.Piecewise((f1_expr, x <= 0), (f2_expr, True))


class TestCase:
    """Один тест: коэффициенты, лямбда, точное решение и производный от
    него правый член g, плюс векторизованные callable-версии всех функций,
    готовые к передаче в ChernovSolver / FiniteDifferenceSolver / ... .
    """

    def __init__(self, name, category, a_expr, b_expr, c_expr, lam, f_expr,
                 x_min=X_MIN, x_max=X_MAX, notes=""):
        self.name = name
        self.category = category
        self.lam = float(lam)
        self.x_min = x_min
        self.x_max = x_max
        self.notes = notes

        g_expr = -(a_expr * sp.diff(f_expr, x, 2)
                   + b_expr * sp.diff(f_expr, x, 1)
                   + (c_expr - lam) * f_expr)

        self.a_expr, self.b_expr, self.c_expr = a_expr, b_expr, c_expr
        self.g_expr, self.f_expr = g_expr, f_expr

        self.a = _to_vectorized(a_expr)
        self.b = _to_vectorized(b_expr)
        self.c = _to_vectorized(c_expr)
        self.g = _to_vectorized(g_expr)
        self.f_exact = _to_vectorized(f_expr)

    def __repr__(self):
        return f"<TestCase {self.category}/{self.name}>"


def build_test_suite():
    """Возвращает список TestCase, покрывающих все категории п.4 task.md."""

    cases = []

    # ──  Постоянные коэффициенты ──────────────────────────────────────
    cases.append(TestCase(
        name="const_baseline", category="constant",
        a_expr=sp.Integer(1), b_expr=sp.Integer(0), c_expr=sp.Integer(0),
        lam=2.0, f_expr=sp.exp(-4 * x**2),
        notes="Чистая диффузия, гауссов колокол, базовая сверка с ядром "
              "резольвенты exp(-sqrt(lam)|x-y|)/(2 sqrt(lam))."))

    cases.append(TestCase(
        name="const_convection", category="constant",
        a_expr=sp.Integer(1), b_expr=sp.Rational(3, 2), c_expr=sp.Rational(1, 2),
        lam=3.0, f_expr=sp.cos(2 * x) * sp.exp(-2 * x**2),
        notes="Диффузия + снос + реакция, немонотонное решение."))

    # ──  Переменные коэффициенты ──────────────────────────────────────
    cases.append(TestCase(
        name="var_smooth", category="variable",
        a_expr=1 + sp.Rational(3, 10) * sp.sin(sp.pi * x),
        b_expr=sp.Rational(2, 5) * sp.cos(sp.pi * x),
        c_expr=sp.Rational(3, 10) * sp.tanh(x),
        lam=1.5, f_expr=sp.exp(-3 * x**2) * sp.sin(2 * x),
        notes="a(x) in [0.7,1.3] — гладкая, но не постоянная эллиптичность. "
              "c(x)=0.3*tanh(x): похоже на 0.3*x вблизи 0 (тот же характер "
              "на [-1,1]), но ограничено на R (|c|<=0.3 всюду) — "
              "требование a,b,c in UC_b(R) из task.md."))

    cases.append(TestCase(
        name="var_pinch", category="variable",
        a_expr=sp.Rational(1, 2) + sp.Rational(2, 5) * sp.cos(2 * sp.pi * x),
        b_expr=sp.Rational(1, 5) * sp.tanh(x),
        c_expr=sp.Integer(0),
        lam=1.0, f_expr=sp.cos(3 * x) * sp.exp(-x**2),
        notes="a(x) in [0.1,0.9] — заметное сужение эллиптичности внутри "
              "окна. b(x)=0.2*tanh(x) вместо 0.2*x — ограничено на R."))

    # ──  Жёсткие задачи (малое a относительно b) ──────────────────────
    cases.append(TestCase(
        name="stiff_const", category="stiffness",
        a_expr=sp.Rational(1, 100), b_expr=sp.Integer(2), c_expr=sp.Integer(0),
        lam=1.0, f_expr=sp.exp(-4 * x**2),
        notes="a/b = 0.005 — то же решение, что const_baseline, но жёсткий "
              "оператор: RK4 должен требовать заметно мельче шаг, чем "
              "FD/Rosenbrock."))

    cases.append(TestCase(
        name="stiff_variable", category="stiffness",
        a_expr=sp.Rational(1, 50) + sp.Rational(1, 100) * sp.cos(sp.pi * x),
        b_expr=3 + sp.sin(sp.pi * x),
        c_expr=sp.Integer(0),
        lam=1.0, f_expr=sp.cos(4 * x) * sp.exp(-2 * x**2),
        notes="a(x) in [0.01,0.03], b(x) in [2,4] — жёсткость меняется вдоль x."))

    # ──  Осцилляции ────────────────────────────────────────────────────
    cases.append(TestCase(
        name="osc_highfreq_solution", category="oscillation",
        a_expr=sp.Integer(1), b_expr=sp.Integer(0), c_expr=sp.Integer(0),
        lam=5.0, f_expr=sp.cos(15 * x) * sp.exp(-x**2),
        notes="Высокочастотное решение — проверка требований к dx."))

    cases.append(TestCase(
        name="osc_coefficient", category="oscillation",
        a_expr=sp.Integer(1), b_expr=sp.Integer(0),
        c_expr=4 * sp.sin(10 * x),
        lam=5.0, f_expr=sp.exp(-2 * x**2),
        notes="Осциллирующий c(x), sup|c|=4 близко к lam=5 (delta=1) — "
              "чувствительность к оценке ||c||_inf и к T."))

    # ──  Склеенные решения ────────────────────────────────────────────
    f1_smooth = sp.exp(-x**2)
    cases.append(TestCase(
        name="glued_smooth", category="glued",
        a_expr=sp.Integer(1), b_expr=sp.Rational(3, 10), c_expr=sp.Rational(1, 5),
        lam=2.0,
        f_expr=glued(f1_smooth, sp.Rational(3, 10) * sp.cos(5 * x), match_order=3),
        notes="При x<=0: exp(-x^2). При x>=0: (полином + x^4*(...)) * "
              "exp(-x^2), склейка гладкая до 3-й производной включительно "
              "(f составное, но глобально ограничено и g остаётся "
              "непрерывной — корректное классическое решение)."))

    cases.append(TestCase(
        name="glued_kink", category="glued",
        a_expr=sp.Integer(1), b_expr=sp.Integer(0), c_expr=sp.Integer(0),
        lam=2.0,
        f_expr=glued(f1_smooth, -sp.Rational(1, 2) * sp.cos(6 * x), match_order=1),
        notes="Склейка только C^1 (значение и первая производная совпадают, "
              "вторая — нет) => g(x) терпит разрыв в x=0. Это ВНЕ гипотез "
              "теоремы (g требуется непрерывной), сознательный стресс-тест "
              "на устойчивость методов за пределами теории, а не "
              "верификационный тест. Хвост именно cos, а не sin: вклад "
              "слагаемого x^2*tail во вторую производную в нуле равен "
              "2*tail(0), и для sin(6x) он обнуляется — склейка тогда "
              "получается C^2, разрыв уезжает в третью производную f, а g "
              "остаётся непрерывной (ровно этот дефект был у первой версии "
              "теста и найден при оценке норм в convergence.ipynb)."))

    return cases


def glued_quasipoly(h_expr, mu, m):
    """Склейка двух квазиполиномов в нуле: e^{mu x} P_m(x) при x<=0 и
    e^{-mu x} Q_m(x) при x>=0.

    P_m и Q_m — отрезки степени m рядов Тейлора e^{-mu x} h(x) и
    e^{mu x} h(x). Тогда обе ветви совпадают с h до x^m включительно, то
    есть склейка C^m, а скачок (m+1)-й производной в общем случае ненулевой.
    Знаки показателей выбраны так, что обе ветви затухают на бесконечности.
    Для чётной h скачок (m+1)-й производной обнуляется и склейка
    получается глаже задуманного, поэтому h берётся нечётной части ненулевой.
    """
    P = sp.expand(sp.series(sp.exp(-mu * x) * h_expr, x, 0, m + 1).removeO())
    Q = sp.expand(sp.series(sp.exp(mu * x) * h_expr, x, 0, m + 1).removeO())
    return sp.Piecewise((sp.exp(mu * x) * P, x <= 0),
                        (sp.exp(-mu * x) * Q, True))


def build_extended_suite():
    """Расширенный набор: по три новые задачи в каждую категорию и две
    склейки квазиполиномов. Статья его не использует, так что
    build_test_suite() и все числа статьи от него не зависят.
    """
    cases = []

    # ── 1. Постоянные коэффициенты ──────────────────────────────────────
    cases.append(TestCase(
        name="const_slow_decay", category="constant",
        a_expr=sp.Rational(1, 2), b_expr=sp.Integer(0), c_expr=sp.Integer(1),
        lam=2.0, f_expr=1 / (1 + x**2),
        notes="Решение и g убывают степенным образом, а не гауссовым: "
              "проверка влияния запаса окна."))
    cases.append(TestCase(
        name="const_negative_c", category="constant",
        a_expr=sp.Integer(2), b_expr=sp.Integer(-1), c_expr=-sp.Rational(1, 2),
        lam=1.0, f_expr=sp.sin(3 * x) * sp.exp(-x**2),
        notes="c < 0 и b < 0; δ = 0.5."))
    cases.append(TestCase(
        name="const_small_lambda", category="constant",
        a_expr=sp.Rational(1, 4), b_expr=sp.Rational(1, 2), c_expr=sp.Integer(0),
        lam=0.5, f_expr=(1 + x) * sp.exp(-x**2 / 2),
        notes="Малое λ = δ = 0.5: длинный интеграл по t и большой δ^{-3}."))

    # ── 2. Переменные коэффициенты ──────────────────────────────────────
    cases.append(TestCase(
        name="var_all_three", category="variable",
        a_expr=1 + sp.tanh(x) / 2, b_expr=sp.sin(x),
        c_expr=sp.cos(2 * x) / 2,
        lam=2.0, f_expr=sp.exp(-x**2) * sp.cos(x),
        notes="Все три коэффициента переменные; a от 0.5 до 1.5."))
    cases.append(TestCase(
        name="var_bump_a", category="variable",
        a_expr=sp.Rational(3, 10) + sp.Rational(7, 10) * sp.exp(-x**2),
        b_expr=-sp.tanh(2 * x) / 2, c_expr=sp.Integer(0),
        lam=1.0, f_expr=x * sp.exp(-x**2),
        notes="a максимально в нуле и падает до 0.3 на краях; b меняет знак "
              "в нуле."))
    cases.append(TestCase(
        name="var_negative_c", category="variable",
        a_expr=1 + sp.cos(3 * x) / 5, b_expr=sp.Rational(3, 10),
        c_expr=-1 + sp.sin(x) / 2,
        lam=2.0, f_expr=sp.exp(-3 * x**2 / 2),
        notes="c переменный и отрицательный, ‖c‖ = 1.5, δ = 0.5."))

    # ── 3. Жёсткие задачи ───────────────────────────────────────────────
    cases.append(TestCase(
        name="stiff_ratio_200", category="stiffness",
        a_expr=sp.Rational(1, 200), b_expr=sp.Integer(1), c_expr=sp.Integer(0),
        lam=1.0, f_expr=sp.exp(-x**2),
        notes="Жёсткость: a/b = 0.005 при b = 1."))
    cases.append(TestCase(
        name="stiff_reverse_drift", category="stiffness",
        a_expr=sp.Rational(1, 50), b_expr=-3 + sp.tanh(x),
        c_expr=sp.Rational(1, 5),
        lam=1.0, f_expr=sp.cos(2 * x) * sp.exp(-x**2),
        notes="Жёсткость при b < 0 (b от −3.76 до −2.24 на [−1,1], "
              "a = 0.02): f на отрезке определяется значениями g левее него, "
              "это проверка запаса окна. В жёстких задачах статьи b > 0."))
    cases.append(TestCase(
        name="stiff_reaction", category="stiffness",
        a_expr=sp.Rational(1, 100), b_expr=sp.Integer(2),
        c_expr=-sp.Rational(1, 2),
        lam=1.0, f_expr=(1 + x / 2) * sp.exp(-4 * x**2),
        notes="Жёсткость (a/b = 0.005) при c ≠ 0; δ = 0.5."))

    # ── 4. Осцилляции ────────────────────────────────────────────────────
    cases.append(TestCase(
        name="osc_mid_solution", category="oscillation",
        a_expr=sp.Integer(1), b_expr=sp.Integer(0), c_expr=sp.Integer(0),
        lam=3.0, f_expr=sp.sin(8 * x) * sp.exp(-x**2),
        notes="Средняя частота решения (8 против 15 в основном наборе)."))
    cases.append(TestCase(
        name="osc_a", category="oscillation",
        a_expr=1 + sp.sin(8 * x) / 2, b_expr=sp.Integer(0),
        c_expr=sp.Integer(0),
        lam=2.0, f_expr=sp.exp(-x**2),
        notes="Осциллирует коэффициент a."))
    cases.append(TestCase(
        name="osc_b", category="oscillation",
        a_expr=sp.Integer(1), b_expr=2 * sp.cos(10 * x), c_expr=sp.Integer(0),
        lam=2.0, f_expr=sp.cos(3 * x) * sp.exp(-x**2),
        notes="Осциллирует коэффициент b."))

    # ── 5. Склеенные решения ────────────────────────────────────────────
    f1 = sp.exp(-x**2)
    cases.append(TestCase(
        name="glued_c2", category="glued",
        a_expr=sp.Integer(1), b_expr=sp.Rational(1, 2), c_expr=sp.Integer(0),
        lam=2.0,
        f_expr=glued(f1, sp.Rational(2, 5) * sp.cos(5 * x), match_order=2),
        notes="Склейка C^2: g непрерывна, g' разрывна."))
    cases.append(TestCase(
        name="glued_c5_var", category="glued",
        a_expr=1 + sp.sin(x) / 5, b_expr=sp.tanh(x) / 5, c_expr=sp.Integer(0),
        lam=2.0,
        f_expr=glued(f1, sp.Rational(1, 2) * sp.cos(4 * x), match_order=5),
        notes="Склейка C^5 при переменных коэффициентах: g ∈ C^3."))
    cases.append(TestCase(
        name="glued_kink_var", category="glued",
        a_expr=1 + sp.cos(x) / 4, b_expr=sp.Rational(1, 2), c_expr=sp.Integer(0),
        lam=2.0,
        f_expr=glued(f1, -sp.Rational(1, 2) * sp.cos(6 * x), match_order=1),
        notes="Склейка C^1 при b ≠ 0 и переменном a: g разрывна."))

    h = (1 + x / 2) * sp.exp(-x**2)
    cases.append(TestCase(
        name="quasipoly_const", category="glued",
        a_expr=sp.Integer(1), b_expr=sp.Rational(1, 2), c_expr=sp.Rational(1, 5),
        lam=2.0, f_expr=glued_quasipoly(h, 3, 7),
        notes="e^{3x}P_7 при x<=0 и e^{-3x}Q_7 при x>=0, склейка C^7: "
              "g ∈ C^5, внутри условий теоремы."))
    cases.append(TestCase(
        name="quasipoly_var", category="glued",
        a_expr=1 + sp.sin(x) / 4, b_expr=sp.tanh(x) / 4,
        c_expr=sp.cos(x) / 5,
        lam=2.0, f_expr=glued_quasipoly(h, 3, 7),
        notes="То же решение, переменные коэффициенты."))

    return cases


def _is_bounded_on_R(expr, directions=(sp.oo, -sp.oo)):
    """True/False, если удалось установить ограниченность |expr(x)| при
    x -> +-oo (символьно, не по сетке — на конечном окне 0.3*x выглядит
    "нормально", неограниченность видна только в пределе); None, если
    sympy не смог вычислить предел (не считаем это ошибкой теста).

    Piecewise разбирается по кускам: sp.limit на самом Piecewise у sympy
    иногда падает на внутренней подстановке x=0 в условие x<=0 при
    x -> -oo; предел -oo фактически определяется только первым куском
    (x<=0), предел +oo — только последним (else/True), поэтому берём
    пределы от их выражений по отдельности.
    """
    folded = sp.piecewise_fold(expr)
    if isinstance(folded, sp.Piecewise):
        # Каждая ветвь проверяется только на своём конце: левая работает
        # лишь при x<=0, и e^{3x}P(x) при x -> +oo склейку не портит.
        left_ok = _is_bounded_on_R(folded.args[0].expr, (-sp.oo,))
        right_ok = _is_bounded_on_R(folded.args[-1].expr, (sp.oo,))
        if left_ok is None or right_ok is None:
            return None
        return left_ok and right_ok

    for direction in directions:
        try:
            lim = sp.limit(sp.Abs(expr), x, direction)
        except Exception:
            return None
        if lim == sp.oo:
            return False
    return True


if __name__ == "__main__":
    print("Проверка a,b,c,g,f_exact на ограниченность на всём R "
          "(условие a,b,c,g in UC_b(R) из task.md, не только на [-1,1]):")
    all_bounded = True
    for tc in build_test_suite():
        for label, expr in [("a", tc.a_expr), ("b", tc.b_expr), ("c", tc.c_expr),
                             ("g", tc.g_expr), ("f_exact", tc.f_expr)]:
            status = _is_bounded_on_R(expr)
            if status is False:
                all_bounded = False
                print(f"  НЕОГРАНИЧЕНО: {tc.name}.{label}(x) = {expr}")
            elif status is None:
                print(f"  не удалось проверить: {tc.name}.{label}(x)")
    print("  все проверенные функции ограничены на R" if all_bounded
          else "  ЕСТЬ ПРОБЛЕМЫ (см. выше)")
    print()

    # Скачки g в точке стыка склеенных тестов. Проверка автоматическая, потому
    # что «класс гладкости» склейки легко получить не тот, который задумывался:
    # у первой версии glued_kink хвост -0.5*sin(6x) давал нулевой вклад во
    # вторую производную в нуле, склейка выходила C^2 вместо C^1, и g
    # оставалась непрерывной — то есть тест не делал того, что заявлено.
    print("Скачки g и её производных в точке стыка x=0 (склеенные тесты):")
    expected_jump = {"glued_smooth": False, "glued_kink": True}
    for tc in build_test_suite():
        folded = sp.piecewise_fold(tc.g_expr)
        if not isinstance(folded, sp.Piecewise):
            continue
        left, right = folded.args[0].expr, folded.args[-1].expr
        jumps = []
        for k in range(3):
            dl = sp.limit(sp.diff(left, x, k), x, 0, '-')
            dr = sp.limit(sp.diff(right, x, k), x, 0, '+')
            jumps.append(float(sp.simplify(dr - dl)))
        has_jump = abs(jumps[0]) > 1e-12
        status = ""
        if tc.name in expected_jump and has_jump != expected_jump[tc.name]:
            status = ("   <-- НЕ СООТВЕТСТВУЕТ ЗАМЫСЛУ ТЕСТА: ожидался "
                      + ("разрыв g" if expected_jump[tc.name] else "непрерывная g"))
        print(f"  {tc.name:14s} скачок g={jumps[0]:+.4g}, g'={jumps[1]:+.4g}, "
              f"g''={jumps[2]:+.4g}{status}")
    print()


    # Проверка независима от _to_vectorized: производные f берутся точно
    # символьно (не численным np.gradient, который у высокочастотных/
    # склеенных случаев сам вносит большую погрешность), и невязка
    # a f'' + b f' + (c-lam) f + g сравнивается через ту же векторизацию,
    # что использует TestCase — так ловятся реальные баги lambdify/
    # broadcast/Piecewise, а не погрешность диагностики.
    suite = build_test_suite()
    xs = np.linspace(X_MIN, X_MAX, 401)
    xs = xs[np.abs(xs) > 1e-9]  # x=0 — точка стыка Piecewise, пропускаем
    print(f"{'name':22s} {'category':11s} {'a range':16s} {'c_sup':8s} "
          f"{'lam-c_sup':10s} {'residual':10s}")
    for tc in suite:
        fpp_expr = sp.diff(tc.f_expr, x, 2)
        fp_expr = sp.diff(tc.f_expr, x, 1)
        fpp = _to_vectorized(fpp_expr)(xs)
        fp = _to_vectorized(fp_expr)(xs)

        a_v, b_v, c_v, g_v, f_v = (tc.a(xs), tc.b(xs), tc.c(xs), tc.g(xs),
                                    tc.f_exact(xs))
        residual = a_v * fpp + b_v * fp + (c_v - tc.lam) * f_v + g_v
        max_res = np.max(np.abs(residual))
        c_sup = np.max(np.abs(c_v))
        print(f"{tc.name:22s} {tc.category:11s} "
              f"[{a_v.min():.3f},{a_v.max():.3f}]   {c_sup:8.3f} "
              f"{tc.lam - c_sup:10.3f} {max_res:10.2e}")
