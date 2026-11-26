import math
from statistics import NormalDist

normal = NormalDist()


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Размер каждой группы для заданного MDE."""
    za = normal.inv_cdf(1 - alpha / (2 * comparisons))
    zb = normal.inv_cdf(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Минимальный обнаруживаемый эффект при n клиентах в каждой группе."""
    za = normal.inv_cdf(1 - alpha / (2 * comparisons))
    zb = normal.inv_cdf(power)
    return (za + zb) * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    """Разность средних, доверительный интервал и p-значение."""
    diff = t.mean() - c.mean()
    se = math.sqrt(t.var() / len(t) + c.var() / len(c))
    q = normal.inv_cdf(1 - alpha / 2)
    p = math.erfc(abs(diff / se) / math.sqrt(2))
    return {"diff": diff, "lo": diff - q * se, "hi": diff + q * se, "p": p}


def srm_pvalue(sizes):
    """Проверка равных долей для двух или трёх групп."""
    expected = sum(sizes) / len(sizes)
    chi2 = sum((size - expected) ** 2 / expected for size in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("Ожидаются размеры двух или трёх групп")


def cuped(y, x):
    """Метрика с поправкой CUPED и коэффициент theta."""
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta
