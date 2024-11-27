# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from kaggle_environments import make, evaluate

# Определяем агентов в классовом формате для удобства использования
class RockAgent:
    def __call__(self, observation, configuration):
        return 0  # всегда выбирает камень

class PaperAgent:
    def __call__(self, observation, configuration):
        return 1  # всегда выбирает бумагу

class ScissorsAgent:
    def __call__(self, observation, configuration):
        return 2  # всегда выбирает ножницы

class CopyAgent:
    def __call__(self, observation, configuration):
        # Повторяет действие противника
        if observation.step > 0:
            return observation.lastOpponentAction  # Сохраняет статус противника с предыдущего хода
        return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде

class ReactionaryAgent:
    def __call__(self, observation, configuration):
        # Реагирует на последнее действие противника
        if observation.step == 0:
            return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде
        return (observation.lastOpponentAction + 1) % configuration.signs  # Выбирает действие, которое бьет последнее действие противника

class ContrReactionaryAgent:
    def __init__(self):
        self.last_step = 0  # Инициализация последнего действия

    def __call__(self, observation, configuration):
        # Каждый ход выбирает действие, противоположное тому, что выбрал противник
        self.last_step = (self.last_step + 2) % configuration.signs  # Противоположный ход
        return self.last_step

class MonotonousAgent:
    def __call__(self, observation, configuration):
        # Выбирает фиксированную последовательность действий по модулю количества знаков
        return observation.step % configuration.signs if observation.step > 0 else random.randrange(0, configuration.signs)

class StatisticalAgent:
    def __init__(self):
        self.action_histogram = {}  # Хранит историю действий противника

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

class MonotonousStepAgent:
    def __init__(self):
        self.last_step = -1  # Инициализация последнего действия

    def __call__(self, observation, configuration):
        self.last_step = self.last_step if self.last_step >= 0 else random.randrange(0, configuration.signs)
        # Увеличиваем на 2, чтобы поменять направление действия
        return (self.last_step + 2) % configuration.signs

class MonotonousRepeatAgent:
    def __init__(self):
        self.last_step = -1
        self.repeats = 0  # Подсчет повторений

    def __call__(self, observation, configuration):
        if self.last_step < 0:
            self.last_step = random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде
        if self.repeats < 1:
            self.repeats += 1
        else:
            self.repeats = 0
            self.last_step = (self.last_step + 1) % configuration.signs  # Увеличиваем на 1, чтобы сменить действие
        return self.last_step

class PartAgent:
    def __init__(self):
        self.last_step = 0
        self.repeats = 0

    def __call__(self, observation, configuration):
        # Действует в определенных партиях, выкладывая одно действие три раза
        if self.repeats < (configuration.get('episodeSteps') / 3):
            self.repeats += 1
        else:
            self.repeats = 0
            self.last_step = (self.last_step + 1) % configuration.signs  # Сменить действия после финального
        return self.last_step

class ComboAgent:
    def __init__(self):
        self.last_step = 0
        self.repeats = 0
        self.action_histogram = {}  # Хранит историю действий противника

    def __call__(self, observation, configuration):
        if observation.step == 0:
            self.action_histogram.clear()  # Очищаем историю в начале
            return random.randrange(0, configuration.signs)  # Случайный выбор в первом раунде

        action = observation.lastOpponentAction
        # Используем стратегии комбинирования
        if self.repeats < (configuration.get('episodeSteps') / 3):
            self.repeats += 1
            return (action + 1) % configuration.signs  # Вызываem действие, бьющее противника
        else:
            self.repeats = 0
            self.action_histogram[action] = self.action_histogram.get(action, 0) + 1
            # Выбираем статистику частоты действий противника
            mode_action = max(self.action_histogram, key=self.action_histogram.get)
            return (mode_action + 1) % configuration.signs

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
    'monotonous_step': MonotonousStepAgent(),
    'repeat': MonotonousRepeatAgent(),
    'part': PartAgent(),
    'combo': ComboAgent(),
}

class Tournament:
    def __init__(self, agents):
        self.agents = agents
        # Инициализация результирующей таблицы для агентов
        self.results = {agent: {'episodes_sum': 0, 'points': 0} for agent in agents}

    def __save_result(self, round_result, left, right):
        left_wins, right_wins = round_result[0]
        # Обновляем сумму выигрышей
        self.results[left]['episodes_sum'] += left_wins
        self.results[right]['episodes_sum'] += right_wins

        # Обновляем очки на основе побед
        if left_wins > right_wins:
            self.results[left]['points'] += 2
        elif left_wins < right_wins:
            self.results[right]['points'] += 2
        else:
            self.results[left]['points'] += 1
            self.results[right]['points'] += 1

    def start(self, episodes):
        # Запускаем турнир для всех агентов
        names = list(self.agents.keys())
        num_agents = len(names)

        for i in range(num_agents - 1):
            for j in range(i + 1, num_agents):
                left, right = names[i], names[j]
                # Оцениваем результаты турнира между двумя агентами
                round_result = evaluate(
                    "rps",
                    [self.agents[left], self.agents[right]],
                    configuration={"episodeSteps": episodes},
                )
                # Сохраняем результаты
                self.__save_result(round_result, left, right)

    def print_result(self):
        # Сортируем и выводим результаты по очкам и сумме выигрышей
        sorted_by_points = sorted(self.results.items(), key=lambda item: (item[1]['points'], item[1]['episodes_sum']), reverse=True)
        sorted_by_episodes = sorted(self.results.items(), key=lambda item: (item[1]['episodes_sum'], item[1]['points']), reverse=True)

        self.__plot_results(sorted_by_points, 'Очки', 'По очкам')
        self.__plot_results(sorted_by_episodes, 'Сумма выигрышей по эпизодам', 'По эпизодам')

    def __plot_results(self, sorted_results, ylabel, title):
        agents, metrics = zip(*sorted_results)
        df = pd.DataFrame({"agent": agents, ylabel: metrics})

        # Визуализация результатов
        plt.figure(figsize=(15, 7))
        sns.barplot(x='agent', y=ylabel, data=df)
        plt.xlabel("Агент", size=16)
        plt.ylabel(ylabel, size=16)
        plt.xticks(rotation=45)
        plt.title(title, size=20)
        plt.show()

# Запуск турнира
tournament = Tournament(agents)
tournament.start(1000)  # Число эпизодов на турнир
tournament.print_result()  # Вывод результатов
