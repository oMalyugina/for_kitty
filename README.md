я выбрала первую задачу, потому что тут нейронная сеть будет наименьшего размера и я смогу её обучить на моем 
компьютере. В остальных задачах, существует вероятность, что мне не хватит времени на обучение. обучаю на cpu. плюс в 
заметке к задаче сказано попытаться по максимуму сократить нейронную сеть.

Так как классов у нас мало : !!!!!перечисли какие (скорее всего пешеход, машна, велосипед, грузовик) и они очень сильно
 отличаются друг от друга - на не нужны огромные сети вроде resnet и inception. обойдемся несколько слойными сверточными. 

итак. задача 1. классифицировать найденные обхекты. (я пока не задумываюсь будет ли использоваться дальше результаты
 классификации и как. к примеру, если мы потом хотим сделать трекинг - нам имеет слысл обучать также признаки с помощью
  энкодера, чтобы потом сравнивать картинки двух расположенных рядом машин на нескольких фреймах и определять, какие 
  картинки, относятся  одному объекту) 

для начала нам нужно вырезать обучающую выборку. причем, так как последний слой у нас будет полносвязный, вырезать 
будем квадраты одинакового размера. можно было бы вырезать произвольные прямоугольники, только если наша архитектура
 состоит только из сверточных слоев - к примеру какой-нибудь кодировщих, раскодировщик

0. скачиваем данные. я использовала left color images of object data set (12 GB) и training labels of 
object data set (5 MB) со страницы http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d 
1. смотрим формат данных и сами данные - ноутбук visualisation_intup_data.ipynb
2. вырезаем позитивные и смотрим их. размер 64*64
3. скармливаем данные препроцессору, который разобьет их на батчи, а такде на трайн и тест
4. строим модель
5. обучаем и тестируем модель

You can find a solution for first task here. I choose the image classification task, because my private laptop is 
old and don't have a lot of power. Second and third task I couldn't do in one week 
(i think to training on cpu top layers for pretrained voxelnet takes one day minimun). And I also don't have a nessecary 
software (pcl, ros) for visualisation. Installation of all this staff can take from 2 hours to 8 hours.

So. Solution for object classification.

firstly, what you can find in my code:
1. visualisation of input data you can find in notebook "visualisation_input_data.ipynb". you have to define paths to 
images and labes from KITTI
2. cutting and resizing objects from image for training you can find in "data_generator.py"
3. there are spliting dataset to train and test, training, computing errors ans errors visualisation in "train.ipynd"
4. and there are models, trained by me, in models.py

You can find a lot of my comments in notebooks and files. Welcome:) 