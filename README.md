# pascal-part-hss
<details>
<summary>Гайд по установке зависимостей</summary>
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
├── evaluate.py # Код что бы протестировать предобученные модели на тестовой части Pascal-part
├── pyproject.toml # Зависимости для питона 
└── ... # Другие файлы
```

## Информация о тренировке

Все эксперементы с тренировками проводились на 30 эпохах ( кроме resnet101_100_epochs, она тернировалась 100 эпох ) трейновой части Pascal-part. С FocalLoss, размером батча в 8 элементов ( 6 для resnet101 ) и lr 1e-3 для классификатора и 1e-4 для бэкбоуна и 1e-4, 1e-5 соответственно после 75000 итераций. 

## Информация о датасете

Так как изображений довольно мало ( 2826 изображений на трейне ), я использовал разные методы аугментации данных.

* torchvision.transforms.functional.hflip с вероятностью 0.5
* torchvision.transforms.functional.rgb_to_grayscale с вероятностью 0.05
* Здесь выполняется одно из отображений
    * torchvision.transforms.functional.equalize с вероятностью 0.25
    * torchvision.transforms.functional.adjust_saturation(...,  saturation_factor=0.5
            ) с вероятностью 0.25
    * torchvision.transforms.functional.adjust_saturation(...,  saturation_factor=2
            ) с вероятностью 0.25

Так же изображения были приведены к (336, 448).

## Результаты

Здесь предстваленны результаты на тренировок разных бэкбоунов.

| Model Type           | Background | Low hand | Torso | Low leg | Head  | Up leg | Up hand | Upper body | Lower body | Body  |
| -------------------- | ---------- | -------- | ----- | ------- | ----- | ------ | ------- | ---------- | ---------- | ----  |
| resnet50             | 0.938      | 0.400    | 0.578 | 0.334   | 0.712 | 0.389  | 0.452   | 0.320      | 0.439      | 0.725 |
| resnet101_100_epochs | 0.941      | 0.416    | 0.585 | 0.339   | 0.715 | 0.385  | 0.471   | 0.316      | 0.431      | 0.730 |
| resnet101            | 0.941      | 0.418    | 0.591 | 0.358   | 0.717 | 0.403  | 0.469   | 0.319      | 0.447      | 0.734 |
| deeplabv3            | 0.937      | 0.398    | 0.575 | 0.335   | 0.714 | 0.374  | 0.462   | 0.324      | 0.432      | 0.729 |
| segformer            | 0.941      | 0.285    | 0.511 | 0.194   | 0.610 | 0.278  | 0.389   | 0.297      | 0.340      | 0.735 |
| dino_backbone        | 0.759 	    | 0.018    | 0.108 | 0.009 	 | 0.092 | 0.021  | 0.030 	| -          | -          | -     |

## Комментарии
### Бэкбоуны
Видно, что модели не сильно различаются по метрикам. Чуть хуже работает resnet50 и dino_backbone. Я предполагаю, что это связанно с небольшим количеством параметров. Я взял самую маленькую модель dino. Мне кажется увеличивая размер модели, можно добиться чуть лучших метрик, но так как разница между моделями не большая, я не думаю, что изменение архитекутры даст большой прирост.

### Данные
Как я уже писал, данных не очень много и увеличение их количества может улучшить метрики. Можно попробовать другие аугментации, повороты, кропы и т.д.

### Анализ ошибок моделей
Посмотрев на картинки в ноутбуке с аналитикой ошибок модели, на которых модель сильнее всего ошибается, можно заметить, что это модели на которых люди занимают очень мало места. В этом случае модель предсказывает просто монотонный фон, как будто бы человека совсем нет на фотографии. Я предлагаю бороться с этим двумя методами:

1. Можно делать сегментация на кусочках картинок и объединять результаты. Например можно делать супер разрешение картинок до большего размера, например 1024 x 1024 и разбить на 16 квадратов 256x256, на которых предсказывать маски. Тогда каждое из предсказаний будет более "приближенно". Таким образом можно так же тренировать модель случайно кропая изображения разного разрешения и получить больше данных на входе.

2. Можно делать предсказание bounding box'а с помощью другой модели ( например YOLO ) и уже на bounding box'е тренировать сегментационную нейрноку. Или использовать такой подход только на инференсе.