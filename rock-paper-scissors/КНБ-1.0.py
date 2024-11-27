# Импорт необходимых библиотек
# NumPy для работы с массивами
import numpy as np
# Pandas для работы с табличными данными
import pandas as pd
# Random для случайного выбора
import random
# Matplotlib и Seaborn для визуализации результатов
import matplotlib.pyplot as plt
import seaborn as sns
# Kaggle Environments для симуляции игры
from kaggle_environments import make, evaluate


# Определяем агентов в классовом формате для удобства использования
# Агенты будут выбирать, что делать в каждой итерации
class RockAgent:
    def __call__(self, observation, configuration):
        return 0  # всегда выбирает камень (0)


class PaperAgent:
    def __call__(self, observation, configuration):
        return 1  # всегда выбирает бумагу (1)


class ScissorsAgent:
    def __call__(self, observation, configuration):
        return 2  # всегда выбирает ножницы (2)


class CopyAgent:
    def __call__(self, observation, configuration):
        # Повторяет действие противника, если это не первый шаг
        if observation.step > 0:
            return observation.lastOpponentAction
        return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде


class ReactionaryAgent:
    def __call__(self, observation, configuration):
        # Реагирует на последнее действие противника
        if observation.step == 0:
            return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде
        # Выбирает действие, которое бьет последнее действие противника
        return (observation.lastOpponentAction + 1) % configuration.signs


class ContrReactionaryAgent:
    def __init__(self):
        self.last_step = 0  # Инициализация последнего действия

    def __call__(self, observation, configuration):
        # Каждый ход выбирает действие, противоположное тому, что выбрал противник
        self.last_step = (self.last_step + 2) % configuration.signs
        return self.last_step


class MonotonousAgent:
    def __call__(self, observation, configuration):
        # Выбирает фиксированное действие по модулю количества знаков
        return observation.step % configuration.signs if observation.step > 0 else random.randrange(0,
                                                                                                    configuration.signs)


class StatisticalAgent:
    def __init__(self):
        self.action_histogram = {}  # Хранит количество действий противника

    def __call__(self, observation, configuration):
        if observation.step == 0:
            self.action_histogram.clear()  # Очищаем историю в начале
            return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде

        action = observation.lastOpponentAction
        # Обновляем историю действий противника
        self.action_histogram[action] = self.action_histogram.get(action, 0) + 1

        # Выбираем наиболее распространенное действие противника и бьем его
        mode_action = max(self.action_histogram, key=self.action_histogram.get)
        return (mode_action + 1) % configuration.signs


class RandomChoiceAgent:
    def __call__(self, observation, configuration):
        # Совершает случайный выбор
        return random.randrange(0, configuration.signs)


# Новые агенты, добавляем больше стратегий:
class AlwaysScissorsAgent:
    def __call__(self, observation, configuration):
        return 2  # всегда выбирает ножницы


class AlwaysRockAgent:
    def __call__(self, observation, configuration):
        return 0  # всегда выбирает камень


class AlwaysPaperAgent:
    def __call__(self, observation, configuration):
        return 1  # всегда выбирает бумагу


class AlternatingAgent:
    def __init__(self):
        self.step = 0  # Инициализация шага для чередования

    def __call__(self, observation, configuration):
        self.step = (self.step + 1) % 3  # Чередует 0, 1, 2
        return self.step  # Возвращает текущее действие


class StreakBreakerAgent:
    def __init__(self):
        self.last_action = None  # Последнее действие
        self.change_action = False  # Флаг для изменения действия

    def __call__(self, observation, configuration):
        if observation.step == 0:
            self.last_action = random.randrange(0, configuration.signs)
            return self.last_action  # Случайный выбор в первом раунде
        if self.change_action:
            self.last_action = (self.last_action + 1) % configuration.signs
            self.change_action = False
        else:
            self.change_action = True
        return self.last_action  # Возвращает последнее действие


class RandomSwitchAgent:
    def __init__(self):
        self.last_action = random.randrange(0, 3)  # Случайный выбор в начале

    def __call__(self, observation, configuration):
        if random.random() < 0.5:
            self.last_action = (self.last_action + 1) % 3  # Случайное переключение
        return self.last_action  # Возвращает последнее действие


class CopyLastMoveAgent:
    def __call__(self, observation, configuration):
        if observation.step > 0:
            return observation.lastOpponentAction  # Повторяет последнее действие противника
        return random.randrange(0, configuration.signs)  # Случайный выбор для первого раунда


class RandomSavvyAgent:
    def __call__(self, observation, configuration):
        if observation.step == 0:
            return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде
        # Случайное действие, чтобы изменять стратегию
        return (observation.lastOpponentAction + random.randint(0, 2)) % configuration.signs


class MajorityVoteAgent:
    def __init__(self):
        self.action_histogram = {}  # Хранит действия противника

    def __call__(self, observation, configuration):
        if observation.step == 0:
            self.action_histogram.clear()  # Очищаем историю
            return random.randrange(0, configuration.signs)  # Случайный выбор в начале

        action = observation.lastOpponentAction
        self.action_histogram[action] = self.action_histogram.get(action, 0) + 1  # Обновляем статистику действий

        # Находит действие, которое выбирал противник чаще всего
        if len(self.action_histogram) > 1:
            mode_action = max(self.action_histogram, key=self.action_histogram.get)
            return (
                               mode_action + 1) % configuration.signs  # Возвращает действие, которое бьет максимально популярное у противника
        return (action + 1) % configuration.signs  # Если одно действие, то показывает против него


class SlowLearnerAgent:
    def __call__(self, observation, configuration):
        # Случайный выбор каждые 5 шагов
        if observation.step == 0 or (observation.step % 5) == 0:
            return random.randrange(0, configuration.signs)
        return (observation.lastOpponentAction + 1) % configuration.signs  # Каждую итерацию бьет последнее действие


# Справочник агентов
agents = {
    'rock': RockAgent(),
    'paper': PaperAgent(),
    'scissors': ScissorsAgent(),
    'copy_opponent': CopyAgent(),
    'reactionary': ReactionaryAgent(),
    'contr_reactionary': ContrReactionaryAgent(),
    'monotonous': MonotonousAgent(),
    'statistical': StatisticalAgent(),
    'random_choice': RandomChoiceAgent(),
    'always_scissors': AlwaysScissorsAgent(),
    'always_rock': AlwaysRockAgent(),
    'always_paper': AlwaysPaperAgent(),
    'alternating': AlternatingAgent(),
    'streak_breaker': StreakBreakerAgent(),
    'random_switch': RandomSwitchAgent(),
    'copy_last_move': CopyLastMoveAgent(),
    'random_savvy': RandomSavvyAgent(),
    'majority_vote': MajorityVoteAgent(),
    'slow_learner': SlowLearnerAgent(),
}


class Tournament:
    def __init__(self, agents):
        self.agents = agents  # Сохраняем список агентов
        # Инициализация результата для каждого агента
        self.results = {agent: {'episodes_sum': 0, 'points': 0} for agent in agents}

    def __save_result(self, round_result, left, right):
        left_wins, right_wins = round_result[0]  # Успехи агентов в раунде
        self.results[left]['episodes_sum'] += left_wins
        self.results[right]['episodes_sum'] += right_wins

        # Обновление очков на основе побед
        if left_wins > right_wins:
            self.results[left]['points'] += 2  # Победа левого агента
        elif left_wins < right_wins:
            self.results[right]['points'] += 2  # Победа правого агента
        else:
            self.results[left]['points'] += 1  # Ничья
            self.results[right]['points'] += 1

    def start(self, episodes):
        # Запускает турнир для всех агентов
        names = list(self.agents.keys())
        num_agents = len(names)

        for i in range(num_agents - 1):
            for j in range(i + 1, num_agents):
                left, right = names[i], names[j]  # Выбор двух агентов для игры
                round_result = evaluate(
                    "rps",  # Оценка игры "Камень, ножницы, бумага"
                    [self.agents[left], self.agents[right]],
                    configuration={"episodeSteps": episodes},  # Количество шагов в эпизоде
                )
                self.__save_result(round_result, left, right)  # Сохранение результата игры

    def print_result(self):
        # Сортировка и вывод результатов по очкам и сумме выигрышей
        sorted_by_points = sorted(self.results.items(), key=lambda item: (item[1]['points'], item[1]['episodes_sum']),
                                  reverse=True)
        sorted_by_episodes = sorted(self.results.items(), key=lambda item: (item[1]['episodes_sum'], item[1]['points']),
                                    reverse=True)

        self.__plot_results(sorted_by_points, 'Очки', 'По очкам')
        self.__plot_results(sorted_by_episodes, 'Сумма выигрышей по эпизодам', 'По эпизодам')

    def __plot_results(self, sorted_results, ylabel, title):
        agents, metrics = zip(*sorted_results)  # Распаковываем отсортированные результаты
        df = pd.DataFrame({"agent": agents, ylabel: metrics})  # Создание DataFrame для визуализации

        # Визуализация результатов
        plt.figure(figsize=(15, 7))
        sns.barplot(x='agent', y=ylabel, data=df)  # Построение графика
        plt.xlabel("Агент", size=16)  # Подпись по оси X
        plt.ylabel(ylabel, size=16)  # Подпись по оси Y
        plt.xticks(rotation=45)  # Поворот меток X
        plt.title(title, size=20)  # Заголовок графика
        plt.show()  # Показать график


# Запуск турнира
tournament = Tournament(agents)  # Создаем экземпляр турнира
tournament.start(1000)  # Запуск турнира на 1000 эпизодов
tournament.print_result()  # Вывод результатов
