# avito-ds-internship-task

Поиск существующих решений выдал статью ["Fast whitespace correction with encoder-only transformers"](https://aclanthology.org/2023.acl-demo.37.pdf), в которой авторы решали аналогичную проблему для английского языка. Протестировав их решение, я пришел к выводу, что для русского языка результаты не впечатляют. Следующим моим решением было дообучить их baseline модель под русский язык.

TODO: вставить краткий обзор статьи, подходы и сравнение.

Для удобного просмотра оставляю ссылку на их открытре решение:
[whitespace-correction](https://github.com/ad-freiburg/whitespace-correction)

## Структура проекта

```txt
.
├── data
├── notebooks
│   ├── analyze_data.ipynb
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── test_existing_solutions.ipynb
├── README.md
└── src
    └── predict.py
```

- `test_existing_solutions.ipynb` - проверка работы существующих решений

- `analyze_data.ipynb` - анализ тестовых данных для формирования выборки для обучения

- `data_preprocessing.ipynb` - сбор и обработка данных для тренировки модели

- `model_training.ipynb` - код для тренировки модели

- `src/predict.py` - скрипт для запуска предсказания

## Источники

1. Bast, Hannah, Matthias Hertel, and Sebastian Walter. "Fast whitespace correction with encoder-only transformers." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). 2023.