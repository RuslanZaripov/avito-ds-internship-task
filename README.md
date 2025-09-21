# avito-ds-internship-task

## Whitespace Correction

## Описание

Поиск существующих решений выдал статью ["Fast whitespace correction with encoder-only transformers"](https://aclanthology.org/2023.acl-demo.37.pdf), в которой авторы решали аналогичную проблему для английского языка. Протестировав их решение, я пришел к выводу, что для русского языка результаты не впечатляют. Следующим моим решением было дообучить их baseline модель под русский язык.

TODO: вставить краткий обзор статьи, подходы и сравнение.

ByT5 is tokenizer-free version of the T5 model designed to works directly on raw UTF-8 bytes. This means it can process any language, more robust to noise like typos, and simpler to use because it doesn’t require a preprocessing pipeline.

Для удобного просмотра оставляю ссылку на их открытре решение:
[whitespace-correction](https://github.com/ad-freiburg/whitespace-correction)

Стек: Python, PyTorch, NumPy, Pandas, Transformers

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

Советую изучать в следующем порядке:

1. `test_existing_solutions.ipynb` - анализ существующих решений
2. `analyze_data.ipynb` - анализ тестовых данных для формирования выборки для обучения
3. `data_preprocessing.ipynb` - сбор и обработка данных для тренировки модели
4. `model_training.ipynb` - код для тренировки модели
5. `src/predict.py` - скрипт для запуска предсказания

## Метрики

## Вычислительные мощности 

Обучение проводилось на платформе Kaggle GPU P100 (GPU Memory 16 GiB), RAM 29 GiB

```txt
Sat Sep 20 23:49:47 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P100-PCIE-16GB           Off |   00000000:00:04.0 Off |                    0 |
| N/A   46C    P0             33W /  250W |   10257MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

## Идеи для улучшения

## Источники

1. Bast, Hannah, Matthias Hertel, and Sebastian Walter. "Fast whitespace correction with encoder-only transformers." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). 2023.