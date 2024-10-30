# pascal-part-hss
<details>
<summary>Dependecies guide</summary>
<br>
Все команды должны быть выполнены из корня репозитория.
<br>
<pre>

**Venv**

Я устанавливал все зависимости в виртуальную среду, ее можно создать используя ( подразумевая, что python уже установлен ):
```bash
python -m venv .venv
```
После этого виртуальную среду можно активировать:
```bash
source .venv/bin/activate
```
**Python dependencies**

Что бы установить зависимости нужно выполнить:
```bash
pip install .
```
</pre>
</details>

## Структура репозитория

```
pascal-part-hss
├── assignment
├── checkpoints # Папки с весами модели, она должна быть создана автоматически.
├── data # Папки с данными из датасета, она так же должна быть создана автоматически.
├── notebooks # Папка с Jupyter нотбуками с экспериментами. 
|    ├── models
|    |   └── model_name.ipynb # Ноутбуки с тренировками моделей.
|    ├── resnet101_analytics.ipynb # Ноутбук с аналитикой ошибок натренированной модели.
|    └── data_exploration.ipynb # Ноутбук с аналитикой датасета.
├── source
|    ├── data/ # Классы с разными версиями Pascal-part датасета
|    ├── modules/ # Типичные модули из классических архитектур e.g. ResNet
|    ├── utils/ # Разные полезные функции
|    └── constants.py # Важные константы
├── pyproject.toml # Зависимости для питона 
└── ... # Другие файлы
```

## Результаты

| Model Type           | Background | Low hand | Torso | Low leg | Head  | Up leg | Up hand | Upper body | Lower body | Body  |
| -------------------- | ---------- | -------- | ----- | ------- | ----- | ------ | ------- | ---------- | ---------- | ----  |
| resnet50             | 0.938      | 0.400    | 0.578 | 0.334   | 0.712 | 0.389  | 0.452   | 0.320      | 0.439      | 0.725 |
| resnet101_100_epochs | 0.941      | 0.416    | 0.585 | 0.339   | 0.715 | 0.385  | 0.471   | 0.316      | 0.431      | 0.730 |
| resnet101            | 0.941      | 0.418    | 0.591 | 0.358   | 0.717 | 0.403  | 0.469   | 0.319      | 0.447      | 0.734 |
| deeplabv3            | 0.937      | 0.398    | 0.575 | 0.335   | 0.714 | 0.374  | 0.462   | 0.324      | 0.432      | 0.729 |
| segformer            | 0.941      | 0.285    | 0.511 | 0.194   | 0.610 | 0.278  | 0.389   | 0.297      | 0.340      | 0.735 |
| dino_backbone        | 0.759 	    | 0.018    | 0.108 | 0.009 	 | 0.092 | 0.021  | 0.030 	| -          | -          | -     |

## Комментарии