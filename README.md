я выбрала первую задачу, потому что тут нейронная сеть будет наименьшего размера и я смогу её обучить на моем компьютере. В остальных задачах, существует вероятность, что мне не хватит времени на обучение. обучаю на cpu. плюс в заметке к задаче сказано попытаться по максимуму сократить нейронную сеть.

Так как классов у нас мало : !!!!!перечисли какие (скорее всего пешеход, машна, велосипед, грузовик) и они очень сильно отличаются друг от друга - на не нужны огромные сети вроде resnet и inception. обойдемся несколько слойными сверточными. 

итак. задача 1. классифицировать найденные обхекты. 

для начала нам нужно вырезать обучающую выборку. причем, так как последний слой у нас будет полносвязный, вырезать будем квадраты одинакового размера. можно было бы вырезать произвольные прямоугольники, только если наша архитектура состоит только из сверточных слоев - к примеру какой-нибудь кодировщих, раскодировщик

1. вырезаем позитивные и негативные примеры и смотрим их.
2. какие примеры берем в негативные - для начала рандомные. а потом те, на которых наша сеть ошиблась