я выбрала первую задачу, потому что тут нейронная сеть будет наименьшего размера и я смогу её обучить на моем компьютере. В остальных задачах, существует вероятность, что мне не хватит времени на обучение. обучаю на cpu. плюс в заметке к задаче сказано попытаться по максимуму сократить нейронную сеть.

Так как классов у нас мало : !!!!!перечисли какие (скорее всего пешеход, машна, велосипед, грузовик) и они очень сильно отличаются друг от друга - на не нужны огромные сети вроде resnet и inception. обойдемся несколько слойными сверточными. 

итак. задача 1. классифицировать найденные обхекты. (я пока не задумываюсь будет ли использоваться дальше результаты классификации и как. к примеру, если мы потом хотим сделать трекинг - нам имеет слысл обучать также признаки с помощью энкодера, чтобы потом сравнивать картинки двух расположенных рядом машин на нескольких фреймах и определять, какие картинки, относятся  одному объекту) 

для начала нам нужно вырезать обучающую выборку. причем, так как последний слой у нас будет полносвязный, вырезать будем квадраты одинакового размера. можно было бы вырезать произвольные прямоугольники, только если наша архитектура состоит только из сверточных слоев - к примеру какой-нибудь кодировщих, раскодировщик

0. скачиваем данные. я использовала left color images of object data set (12 GB) и training labels of object data set (5 MB) со страницы http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d 
1. смотрим формат данных и сами данные - ноутбук visualisation_intup_data.ipynb
2. вырезаем позитивные и негативные примеры и смотрим их. размер 128*128
   какие примеры берем в негативные - для начала рандомные. а потом те, на которых наша сеть ошиблась