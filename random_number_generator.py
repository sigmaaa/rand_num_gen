import time
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
import math

# Отримання системного часу в мілісекундах
current_time = int(time.time() * 1000)


class RandomNumberGenerator:
    def __init__(self) -> None:
        self.a = 7**5
        self.b = self.a * current_time % (2**31)

    def generate_random_number(self):
        self.b = (self.a * self.b) % (2**31)
        return self.b * 2**-31

    def generate_rayleigh(self, sigma=1.0):
        u = self.generate_random_number()
        return sigma * math.sqrt(-2 * math.log(1 - u))


def test_aperiodicity(sequence, max_period=100):
    # Перевірка чи є період в межах max_period
    for period in range(1, max_period + 1):
        is_periodic = all(sequence[i] == sequence[i + period]
                          for i in range(len(sequence) - period))
        if is_periodic:
            return f"Послідовність має період {period}"
    return "Послідовність аперіодична"


def test_moments(sequence):
    mean = np.mean(sequence)
    variance = np.var(sequence)

    expected_mean = 0.5
    expected_variance = 1/12

    mean_diff = abs(mean - expected_mean)
    variance_diff = abs(variance - expected_variance)

    return {
        "Observed Mean": mean,
        "Expected Mean": expected_mean,
        "Mean Difference": mean_diff,
        "Observed Variance": variance,
        "Expected Variance": expected_variance,
        "Variance Difference": variance_diff
    }


def test_covariance(sequence):
    n = len(sequence)
    if n < 2:
        return "Послідовність занадто коротка для тесту на коваріацію"

    mean = np.mean(sequence)
    covariance = np.mean(
        [(sequence[i] - mean) * (sequence[i + 1] - mean) for i in range(n - 1)])

    return {
        "Covariance": covariance,
        # Для випадкової послідовності коваріація повинна бути близька до 0
        "Expected Covariance": 0,
    }


def kolmogorov_smirnov_test(sequence):
    # Перевірка на рівномірний розподіл [0, 1]
    test_statistic, p_value = kstest(sequence, 'uniform')

    return {
        "KS Test Statistic": test_statistic,
        "p-value": p_value,
        "Conclusion": "Розподіл відповідає рівномірному" if p_value > 0.05 else "Розподіл не відповідає рівномірному"
    }


def plot_aperiodicity(sequence, max_period=100):
    correlation_values = []

    # Перевірка кореляції для різних періодів
    for period in range(1, max_period + 1):
        correlation = sum(sequence[i] == sequence[i + period]
                          for i in range(len(sequence) - period))
        correlation_values.append(correlation)

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_period + 1), correlation_values, marker='o')
    plt.title('Тест на аперіодичність: Кореляція для різних періодів')
    plt.xlabel('Період')
    plt.ylabel('Кількість збігів')
    plt.grid(True)
    plt.show()


def plot_moments(sequence):
    plt.figure(figsize=(10, 6))

    # Гістограма послідовності
    plt.hist(sequence, bins=30, density=True, alpha=0.6,
             color='g', label='Емпіричний розподіл')

    # Теоретичний розподіл для рівномірного розподілу
    plt.axhline(y=1, color='r', linestyle='--',
                label='Теоретичний рівномірний розподіл')

    plt.title('Тест на збіг моментів: Гістограма розподілу')
    plt.xlabel('Випадкове значення')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_covariance(sequence):
    plt.figure(figsize=(10, 6))

    # Побудова графіка розсіювання для сусідніх значень
    plt.scatter(sequence[:-1], sequence[1:], alpha=0.6)

    plt.title('Тест на коваріацію: Графік розсіювання сусідніх значень')
    plt.xlabel('Значення X[i]')
    plt.ylabel('Значення X[i+1]')
    plt.grid(True)
    plt.show()


def plot_kolmogorov_smirnov(sequence):
    # Сортуємо послідовність для емпіричного розподілу
    sorted_sequence = sorted(sequence)
    n = len(sorted_sequence)

    # Кумулятивна функція розподілу (CDF) для емпіричних даних
    empirical_cdf = [i / n for i in range(1, n + 1)]

    plt.figure(figsize=(10, 6))

    # Побудова CDF для емпіричного розподілу
    plt.step(sorted_sequence, empirical_cdf, label='Емпіричний CDF', color='b')

    # Побудова CDF для теоретичного розподілу
    plt.plot([0, 1], [0, 1], label='Теоретичний рівномірний CDF',
             color='r', linestyle='--')

    plt.title('Критерій Колмогорова-Смірнова: Порівняння CDF')
    plt.xlabel('Випадкове значення')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()


rng = RandomNumberGenerator()
sequence = [rng.generate_random_number()
            for _ in range(1000)]  # Генерація 1000 випадкових чисел

# Запуск тестів
print(test_aperiodicity(sequence))
print(test_moments(sequence))
print(test_covariance(sequence))
print(kolmogorov_smirnov_test(sequence))
plot_aperiodicity(sequence)
plot_moments(sequence)
plot_covariance(sequence)
plot_kolmogorov_smirnov(sequence)
# Функція для оцінки параметра σ за вибіркою


def estimate_sigma(data):
    # Оцінка σ для розподілу Релея: σ = sqrt( (1/2) * E[X^2] )
    mean_squared = np.mean(np.array(data)**2)
    estimated_sigma = math.sqrt(mean_squared / 2)
    return estimated_sigma


# Ініціалізація генератора випадкових чисел
rng = RandomNumberGenerator()

# Генеруємо вибірку випадкових чисел за розподілом Релея
sigma_true = 1.0  # Істинне значення σ
sample_size = 1000  # Розмір вибірки
rayleigh_data = [rng.generate_rayleigh(sigma_true) for _ in range(sample_size)]

# 1. Перевірка критерію Колмогорова-Смірнова
# Виконуємо тест Колмогорова-Смірнова з теоретичним розподілом Релея
D, p_value = kstest(rayleigh_data, 'rayleigh', args=(0, sigma_true))
print(f"Kolmogorov-Smirnov test result: D = {D}, p-value = {p_value}")

# 2. Оцінка параметра σ та перевірка гіпотези про збіг
estimated_sigma = estimate_sigma(rayleigh_data)
print(f"True σ: {sigma_true}, Estimated σ: {estimated_sigma}")

# 3. Побудова гістограми та графіків
plt.figure(figsize=(12, 6))

# Емпірична гістограма
plt.subplot(1, 2, 1)
plt.hist(rayleigh_data, bins=30, density=True, alpha=0.6, color='g')
plt.title('Empirical Histogram of Rayleigh Distribution')
plt.xlabel('Value')
plt.ylabel('Normalized Frequency')

# Теоретична крива розподілу Релея
x = np.linspace(0, max(rayleigh_data), 1000)
theoretical_pdf = (x / (sigma_true**2)) * np.exp(-x**2 / (2 * sigma_true**2))
plt.plot(x, theoretical_pdf, 'r', linewidth=2, label='Theoretical PDF')
plt.legend()

# Порівняння емпіричної та теоретичної CDF
plt.subplot(1, 2, 2)
sorted_data = np.sort(rayleigh_data)
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
theoretical_cdf = 1 - np.exp(-sorted_data**2 / (2 * sigma_true**2))

plt.plot(sorted_data, empirical_cdf, label='Empirical CDF', color='b')
plt.plot(sorted_data, theoretical_cdf, label='Theoretical CDF',
         color='r', linestyle='dashed')
plt.title('Empirical vs Theoretical CDF (Rayleigh)')
plt.xlabel('Value')
plt.ylabel('CDF')
plt.legend()

plt.tight_layout()
plt.show()
