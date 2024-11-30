import pandas as pd

# Configure pandas to print all rows and columns
pd.options.display.max_rows = 50
pd.options.display.max_columns = None

ID_QUESTIONS = 0
ID_LECTURES = 1


if __name__ == "__main__":
    # Создадим датафреймы из исходных файлов
    lectures_table = pd.read_csv("./ext_files/lectures.csv")
    questions_table = pd.read_csv("./ext_files/questions.csv")
    train_table = pd.read_csv("./ext_files/train-001.csv")

    # Изменим тип prior_questuin_had_explanation на float
    train_table["prior_question_had_explanation"] = train_table.prior_question_had_explanation.astype(float)

    # Создадим 2 датафрейма из тренировочных данных: лекции и вопросы

    # Используем dropna() для "вопросов",
    # исходя из описания для prior_question_had_explanation, где указано,
    # что это вопросы без обратной связи, используемые для диагностики.
    train_questions = train_table[train_table.content_type_id == ID_QUESTIONS].dropna()


    # Уберём лишние колонки из train таблицы для лекций.
    not_required_columns = [
        "prior_question_had_explanation",
        "prior_question_elapsed_time", "answered_correctly",
        "user_answer", "timestamp"
    ]
    train_lectures = train_table[
        train_table.content_type_id == ID_LECTURES
    ].drop(
        not_required_columns, axis=1
    )

    # Проанализуем полное описание полученной таблицы
    main_columns = [
        "timestamp", "answered_correctly",
        "prior_question_elapsed_time",
        "prior_question_had_explanation"
    ]
    print(train_questions.filter(main_columns).describe())
    # Сразу видно, что высокими являются
    # процент правильности ответов и
    # процент просмотра пояснений к предыдущей группе вопросов

    # region Зависимость правильности ответов от времени, затраченного на ответ
    print(
        train_questions.groupby(
            ["answered_correctly"], as_index=False
        ).aggregate(
            {'prior_question_elapsed_time': 'mean'}
        ).sort_values(
            "answered_correctly"
        )
    )
    # Явной корреляции не наблюдается
    # endregion

    # region Зависимость правильности ответа от чтения пояснений к вопросам
    print(train_questions.filter(
        ["prior_question_had_explanation", "answered_correctly"]
    ).groupby(
        'prior_question_had_explanation'
    ).aggregate(
        {'answered_correctly': 'mean'}
    ))
    # Вывод: Пользователь, прочитавший пояснение к предыдущему блоку вопросов,
    # с большей вероятность верно ответит на следующий вопрос

    # Проверим эту теорию про чтение к текущим вопросами
    train_questions["current_question_explained"] = train_questions.prior_question_had_explanation.shift(1)
    print(
        train_questions.filter(
            ["current_question_explained", "answered_correctly"]
        ).groupby(
            'current_question_explained'
        ).aggregate(
            {'answered_correctly': 'mean'}
        )
    )
    # Вывод: После правильного ответа пользователи чаще смотрят пояснения, чем после неправильного
    # endregion

    # region Зависимость правильности ответов пользователя от его посещаемости лекций
    lectures_count = len(lectures_table.lecture_id)

    # Получим общую таблицу train и lectures
    # для анализа посещаемости и успешности тестирования
    train_lectures_inner_join = train_lectures.merge(
        lectures_table, left_on="content_id", right_on="lecture_id"
    )
    # получим количество посещённых лекций для каждого пользователя
    listened_lectures = train_lectures_inner_join.groupby(
        "user_id", as_index=False
    ).aggregate(
        {'lecture_id': 'count'}
    ).rename(
        columns={'lecture_id': 'listened_lectures_perc'}
    ).sort_values(
        "listened_lectures_perc"
    )
    listened_lectures["listened_lectures_perc"] = (
        listened_lectures["listened_lectures_perc"].
        apply(lambda x: x/lectures_count)
    )
    # Получим таблицу успешности пользователей
    users_success = train_questions.groupby(
        "user_id", as_index=False
    ).aggregate(
        {'answered_correctly': 'mean'}
    ).rename(
        columns={'answered_correctly': 'answered_correctly_perc'}
    ).sort_values(
        "answered_correctly_perc"
    )
    # Объединим две предыдущие таблицы и получим зависимость успешности тестирования от посещаемости лекций
    print(
        listened_lectures.merge(
            users_success, on="user_id"
        )
    )

    # Вывод: Успешность тестирования напрямую зависит от посещаемости лекций
    # endregion

    # region Зависимость правильности ответа от bundle_id/part/tag
    # Далее проверим следующие взаимосвязи:
    # Зависимость правильности ответов от bundle_id вопроса
    # Зависимость правильности ответов от part вопроса
    # Зависимость правильности ответов от tag вопроса

    # Сгруппируем все ответы по ID и выведем % правильности ответов.
    answers_grouped_by_questions = train_questions.groupby(
        "content_id", as_index=False
    ).aggregate(
        {'answered_correctly': 'mean'}
    )
    # Сгруппируем все ответы по ID и получим % правильности ответов.
    # Объединим полученную таблицу с таблицей описания вопросов.
    questions_full_info = train_questions.groupby(
        "content_id", as_index=False
    ).aggregate(
        {'answered_correctly': 'mean'}
    ).merge(
        questions_table, left_on=["content_id"], right_on=["question_id"]
    )

    # Правильность ответов в зависимости от bundle_id вопроса
    print(
        train_questions.groupby(
            "content_id", as_index=False
        ).aggregate(
            {'answered_correctly': 'mean'}
        ).merge(
            questions_table, left_on=["content_id"], right_on=["question_id"]
        ).groupby(
            "bundle_id", as_index=False
        ).aggregate(
            {'answered_correctly': 'mean'}
        ).sort_values(
            "answered_correctly"
        )
    )
    # Вывод: есть слишком сложные бандлы вопросов, а есть слишком простые.
    # Перераспределение тестов по бандлам позволило бы избежать появления слишком сложных и слишком легких бандлов
    # PS. Если я верно понимаю, что bundle это что-то вроде варианта на тестированиях.

    # Правильность ответов в зависимости от part вопроса в рамках теста
    print(
        questions_full_info.groupby(
            "part", as_index=False
        ).aggregate(
            {'answered_correctly': 'mean'}
        ).sort_values(
            "answered_correctly"
        )
    )
    # Вывод: Известно какие части тестирования могут вызвать самые большие сложности,
    # им стоит уделять больше времени

    # Зависимость правильности ответов от tag вопросов.
    questions_full_info["tags"] =  questions_full_info["tags"].str.split(' ')
    print(
        questions_full_info.filter(
            ["tags", "answered_correctly"]
        ).explode(
            "tags"
        ).groupby(
            "tags", as_index=False
        ).aggregate(
            {'answered_correctly': 'mean'}
        ).sort_values(
            "answered_correctly"
        )
    )
    # Вывод: Получены теги в градации их сложности для пользователей
    # endregion
