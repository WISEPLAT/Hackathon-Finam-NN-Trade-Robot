# Торговый робот с использованием нейросетей
## Hackathon-Finam-NN-Trade-Robot
```shell
Сделан в рамках соревнования Хакатон «Финам Trade API» 
по созданию торговых систем на основе открытого торгового API «Финама»
```

#### Почему я выбрал использование нейросетей для торгового робота?
1. Тема использования искусственного интеллекта актуальна:
   - для прогнозирования поведения фондового рынка в целом, 
   - для осуществления предсказаний поведения цены отдельных акций и/или фьючерсов и других инструментов
   - для поиска определенных торговых формаций на графиках цен
    

2. Широкое применение искусственного интеллекта очень активно развивается на Западных рынках, на Российском всё только начинается.
   

3. В открытом доступе нет полноценных примеров по использованию нейросетей для прогнозирования цен акций/фьючерсов, а те которые есть
   - или не работают 
   - или чего-то для их работы постоянно не хватает.
     - По крайней мере лично мне ещё ни разу не встретились полноценно работающие примеры.


Поэтому и принял решение сделать торгового робота, который использует нейросети на основе компьютерного зрения для поиска определенных формаций 
на торговом графике акций и используя лучшую обученную модель осуществляет торговые операции.  

#### Какие есть скрытые цели? )))
Т.к. этот пример торгового робота с использованием нейросетей хорошо документирован и последовательно проходит через все этапы:
    
- получение исторических данных по акциям
- подготовка датасета с картинками формаций из графика акций по определенной логике
- обучение нейросети и выбор лучшей обученной модели по параметрам loss, accuracy, val_loss, val_accuracy 
- проверка предсказаний сделанных нейросетью
- проверка подключения к API Финама
- запуск live стратегии с использованием выбранной лучшей модели обученной нейросети
- записано обучающее видео как запускать и работать с этим кодом, выложенное [на YouTube](https://youtu.be/yrQFqvc4fk0 ) и [на RuTube](https://rutube.ru/video/private/1255dfe65f4db8736b894cae72b14c45/?p=oOFSDPr1El6lq586tIm2qg )

то, это позволит всем, кто только начинает свой путь по применению нейросетей для аналитики, использовать этот код, 
как стартовый шаблон с последующим его усовершенствованием и допиливанием)) 

- По крайней мере появился +1 рабочий пример использования нейросетей для аналитики цен графика акций.

Тем самым, станет больше роботов с использованием искусственного интеллекта,
```
- это повлечет большую волатильность нашего фондового рынка
- большую ликвидность за счет большего количества сделок
- и соответственно больший приток капитала в фондовый рынок
```

#### Зарабатывает ли сейчас этот робот?
Торговая стратегия заложенная в этом роботе не даст ему заработать, 
т.к. мы открываем позицию по подтвержденной нейросетью формации на графике 
(т.е. когда нейросеть предсказывает, что вероятно будет рост), 
но мы не ждем роста и закрываем позицию через +1 бар старшего таймфрейма.
*Как вариант, можно заходить в сделку 1 к 3 или 1 к 5 со стоп-лоссом. Т.е. ждать профита или стоп-лосса.))*

==========================================================================

## Установка
1) Самый простой способ:
```shell
git clone https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot
```

2) Или через PyCharm:
- нажмите на кнопку **Get from VCS**:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/get_from_vcs.jpg )

Вот ссылка на этот проект:
```shell
https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot
```

- вставьте эту ссылку в поле **URL** и нажмите на кнопку **Clone** 
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/paste_url_push_clone.jpg)


- Теперь у нас появился проект торгового робота:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/hackathon_finam_nn_trade_robot.jpg )

### Установка дополнительных библиотек
Для работы торгового робота с использованием нейросетей, есть некоторые библиотеки, которые вам необходимо установить:
```shell
pip install aiohttp aiomoex pandas matplotlib tensorflow finam-trade-api
```

так же их можно установить такой командой
```shell
pip install -r requirements.txt
```

Обязательно! Выполните в корне вашего проекта через терминал эту команду:
```shell
git clone https://github.com/cia76/FinamPy
```
для клонирования библиотеки, которая позволяет работать с функционалом API брокера Финам.

P.S. Библиотека finam-trade-api - тоже позволяет работать с API Финам, просто для тестов я использовал обе.))) А для live торговли FinamPy.

Теперь наш проект выглядит вот так:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/hackathon_finam_nn_trade_robot_add_lib.jpg )

### Начало работы

Вот перечень задач, которые нужно сделать для успешного запуска торгового робота использующего нейросети на основе компьютерного зрения для поиска формаций 
на торговом графике акций и осуществления им торговых операций:


1. Настроить конфигурационный файл my_config\trade_config.py

   - В нём можно указать по каким тикерам ищем формации и обучаем нейросеть (**training_NN**), и так же указать по каким тикерам торгуем (**portfolio**) используя обученную нейросеть. Остальные параметры можно оставить как есть.
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/trade_config.png )
   

2. Нужно получить исторические данные по акциям, для обучения нейросети
   - Исторические данные для обучения нейросети мы получаем с MOEX. Т.к. получаем их бесплатно, то есть задержка в полученных данных на 15 минут.
   - Для этого используется файл **1_get_historical_data_for_strategy_from_moex.py**
   - Полученные исторические данные сохраняются в каталоге **csv** в CSV файлах.
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/historical_data.png )
   

3. Когда есть исторические данные, теперь мы можем подготовить картинки для обучающего набора данных
   - подготовка датасета с картинками формаций из графика акций по определенной логике:
     - на картинке рисуется цена закрытия и две скользящие средние - картинки рисуются для младшего таймфрейма
     - если на старшем таймфрейме закрытие выше предыдущего закрытия, то такой картинке назначаем класс **1** иначе **0**
   - Для этого используется файл **2_prepare_dataset_images_from_historical_data.py**
   - Полученные картинки сохраняются в каталоге **NN\training_dataset_M1** в подкаталогах классификаций **0** и **1**.
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/neural_network_training.png )


4. Наконец-то есть датасеты для обучения нейросети )) Теперь обучаем нейросеть 
   - Используем сверточную нейронную сеть (CNN)
   - Для этого используется файл **3_train_neural_network.py**
   - Лог обучения нейросети находится в файле **3_results_of_training_neural_network.txt**
   - Сходимость нейросети находится в файле **3_Training and Validation Accuracy and Loss.png**
   - При обучении нейросети файлы моделей сохраняются в каталог **NN\\_models** 
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/training.png )


5. После успешного обучения нейросети нужно выбрать одну из обученных моделей для нашего торгового робота
   - Выбор лучшей обученной модели происходит по параметрам loss, accuracy, val_loss, val_accuracy
   - Выбранную модель нужно **вручную** сохранить в каталог **NN_winner** под именем **cnn_Open.hdf5**
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/accuracy_loss.png )


6. Теперь нужно сделать проверку предсказаний сделанных нейросетью на части классифицированных картинках
   - Для этого используется файл **4_check_predictions_by_neural_network.py**
   - Как говорится просто проверить - что Ок
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/classification.png )


7. Наконец-то делаем проверку подключения к API Финама, чтобы мы смогли торговать
   - Для этого используется файл **5_test_api_finam_v1.py** - используем [FinamPy](https://github.com/cia76/FinamPy ) для тестов
   - и файл **6_test_api_finam_v2.py** - используем [FinamTradeApiPy](https://github.com/DBoyara/FinamTradeApiPy ) для тестов

      ####  Как получить токен API Финам:
      - Открыть счет в "Финаме" https://open.finam.ru/registration
      - Зарегистрироваться в сервисе Comon https://www.comon.ru/
      - В личном кабинете Comon получить токен https://www.comon.ru/my/trade-api/tokens для выбранного торгового счета
      - Скопируйте и вставьте в файл **my_config\Config.py** полученный **Ключ API** и **номер торгового счета** (пример конфиг файла здесь: **my_config\Config_example.py**)
      
      ```python
      # content of my_config\Config.py 
      class Config:
          ClientIds = ('<Торговый счет>',)  # Торговые счёта
          AccessToken = '<Токен>'  # Торговый токен доступа
      ```


8. Теперь мы готовы запустить торгового робота в live режиме
      - Не забываем про **Ключ API** и **номер торгового счета**, уже должны быть прописаны в файле **my_config\Config.py**
      - запуск live стратегии осуществляется с помощью файла **7_live_strategy.py**
      - в строке 266 этот параметр **days_back** отвечает за сколько дней назад взять данные, если запускаете скрипт в понедельник или после выходного/праздничного дня, то увеличьте это значение
   
          ```days_back = 1  # на сколько дней назад берем данные```

      - строку 206 можно раскомментировать, чтобы скрипт например не запускался, если рынок не открыт (выходные и праздники не учитывает)
   
          ```await self.ensure_market_open()  # проверяем, что рынок открыт```

      - конфигурация торгового робота находится в файле **my_config\trade_config.py** 
   хотелось бы указать, что т.к. мы берем исторические данные из MOEX, а не по API Финам (т.к. такой функционал будет реализован позже), то доступные тикеры для аналитики и скачивания данных необходимо подбирать вручную.
          ```
            training_NN = {"SBER", "VTBR"}  # тикеры по которым обучаем нейросеть
            portfolio = {"SBER", "VTBR"}  # тикеры по которым торгуем и скачиваем исторические данные
           ```
      - лог запуска торгового робота в live режиме торгов находится в файле **7_live_trading_log.txt** 
      
      запущенный в live режиме торговый робот и торговый терминал:
      ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/live_orders_2_r.jpg )
      
      ордера выставленные торговым роботом:
      ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/live_orders_r.jpg )


Теперь можно запускать и смотреть, а предварительно лучше посмотреть видео по работе с этим кодом
, выложенное [на YouTube](https://youtu.be/yrQFqvc4fk0 ) и [на RuTube](https://rutube.ru/video/private/1255dfe65f4db8736b894cae72b14c45/?p=oOFSDPr1El6lq586tIm2qg )

### Внимание
Некоторые файлы содержат строку:
```exit(777)  # для запрета запуска кода, иначе перепишет результаты```
это сделано специально, чтобы случайно не перезаписать данные, её можно закомментировать, когда будете тестировать свои модели и свои настройки.


P.S. В коде стратегии не реализована проверка на доступность денежных средств на счете для входа в сделку.

Код тестировался на ```M1=>M10``` и ```M10=>H1```, для других таймфреймов необходимо создавать большее число обучающих выборок.

Работоспособность проверялась на ```Python 3.10+``` и ```Python 3.11+``` с последними версиями библиотек.


==========================================================================

## Спасибо
- [FinamPy](https://github.com/cia76/FinamPy ): Игорю за библиотеку, которая позволяет работать с функционалом API брокера Финам.
- [FinamTradeApiPy](https://github.com/DBoyara/FinamTradeApiPy ): DBoyara за библиотеку, асинхронного REST-клиента для API Finam.
- tensorflow: За простую и классную библиотеку для работы с нейросетями.
- aiomoex: За хорошую реализацию получения данных с moex.

## Важно
Исправление ошибок, доработка и развитие кода осуществляется автором и сообществом!

**Пушьте ваши коммиты!** 

# Условия использования
Программный код выложенный по адресу https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot в сети интернет, позволяющий совершать торговые операции на фондовом рынке с использованием нейросетей - это **Программа** созданная исключительно для удобства работы и изучения применений нейросетей/искусственного интеллекта.
При использовании **Программы** Пользователь обязан соблюдать положения действующего законодательства Российской Федерации или своей страны.
Использование **Программы** предлагается по принципу «Как есть» («AS IS»). Никаких гарантий, как устных, так и письменных не прилагается и не предусматривается.
Автор и сообщество не дает гарантии, что все ошибки **Программы** были устранены, соответственно автор и сообщество не несет никакой ответственности за
последствия использования **Программы**, включая, но, не ограничиваясь любым ущербом оборудованию, компьютерам, мобильным устройствам, 
программному обеспечению Пользователя вызванным или связанным с использованием **Программы**, а также за любые финансовые потери,
понесенные Пользователем в результате использования **Программы**.
Никто не ответственен за потерю данных, убытки, ущерб, включаю случайный или косвенный, упущенную выгоду, потерю доходов или любые другие потери,
связанные с использованием **Программы**.

**Программа** распространяется на условиях лицензии [MIT](https://choosealicense.com/licenses/mit).

==========================================================================

----------------------------- English section ----------------------------

# Trading robot using neural networks
## Hackathon-Finam-NN-Trade-Robot
```shell
Created for the «Finam Trade API» Hackathon competition
to create trading systems based on the «Finam Trade API»
```

#### Why did I choose to use neural networks for a trading robot?
1. The topic of using artificial intelligence is relevant:
   - to predict the behavior of the stock market as a whole, 
   - to make predictions about the price behavior of individual stocks and/or futures and other instruments
   - to search for specific trading formations on price charts
    

2. The widespread use of artificial intelligence is very actively developing in the Western markets, in the Russian one everything is just beginning.
   

3. There are no full-fledged examples of using neural networks to predict stock / futures prices in the public, and those that are
   - not working
   - or something for their work is constantly missing.
     - At least personally, I have never met fully working examples.


Therefore, I decided to make a trading robot that uses neural networks based on computer vision to search for certain formations 
on the stock trading chart and uses the best trained model to carry out trading operations..  

#### What are the hidden goals? )))
Because this example of a trading robot using neural networks is well documented and goes through all the steps sequentially:
    
- getting historical data on stocks
- preparation of a dataset with pictures of formations from a stock chart according to a certain logic
- neural network training and selection of the best trained model in terms of loss, accuracy, val_loss, val_accuracy parameters 
- checking the predictions made by the neural network
- checking connection to Finam API
- running a live strategy using the selected best trained neural network model
- a training video was recorded on how to run and work with this code, posted [on YouTube](https://youtu.be/yrQFqvc4fk0 ) and [on RuTube](https://rutube.ru/video/private/1255dfe65f4db8736b894cae72b14c45/?p=oOFSDPr1El6lq586tIm2qg )

then, this will allow everyone who is just starting their journey in using neural networks for analytics to use this code 
as a starting template with its subsequent improvement and completion)) 

- At least +1 working example of using neural networks for stock chart price analytics appeared.

Thus, there will be more robots using artificial intelligence,
```
- this will lead to greater volatility of our stock market
- greater liquidity due to more transactions
- and, accordingly, a greater inflow of capital into the stock market
```

#### Is this robot making money now?
The trading strategy embedded in this robot will not allow it to earn, 
because we open a position on a formation confirmed by the neural network on the chart 
(i.e. when the neural network predicts that there will probably be an increase), 
but we do not wait an increase and close the position after +1 bar of the higher timeframe.
*Alternatively, you can enter a trade with 1 to 3 or 1 to 5 with a stop loss. And wait for profit or stop loss.))*

==========================================================================

## Installation
1) The easiest way:
```shell
git clone https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot
```

2) Or via PyCharm:
- click on the **Get from VCS** button:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/get_from_vcs.jpg )

Here is a link to this project:
```shell
https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot
```

- paste this link into the **URL** field and click on the **Clone** button
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/paste_url_push_clone.jpg)


- Now we have a trading robot project:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/hackathon_finam_nn_trade_robot.jpg )

### Installing Additional Libraries
For the trading robot to work using neural networks, there are some libraries that you need to install:
```shell
pip install aiohttp aiomoex pandas matplotlib tensorflow finam-trade-api
```

they can also be installed with the following command
```shell
pip install -r requirements.txt
```

Necessarily! Execute this command in the root of your project via terminal:
```shell
git clone https://github.com/cia76/FinamPy
```
to clone a library that allows you to work with the Finam broker API functionality.

P.S. The finam-trade-api library also allows you to work with the Finam API, I just used both for tests.))) And for live trading FinamPy.

Now our project looks like this:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/hackathon_finam_nn_trade_robot_add_lib.jpg )

### Beginning of work

Here is a list of tasks that need to be done to successfully launch a trading robot that uses neural networks based on computer vision 
to search for formations on the stock trading chart and carry out trading operations:


1. Customize the configuration file my_config\trade_config.py

   - In it, you can specify which tickers we are looking for formations and train the neural network (**training_NN**), and also indicate which tickers we trade (**portfolio**) using the trained neural network. The rest of the parameters can be left as is.
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/trade_config.png )
   

2. Need to get historical data on stocks to train the neural network
   - We get historical data for training the neural network from MOEX. Because we get them for free, that is, a delay in the received data by 15 minutes.
   - To do this, use the file **1_get_historical_data_for_strategy_from_moex.py**
   - The received historical data is stored in the **csv** directory in CSV files.
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/historical_data.png )
   

3. When there is historical data, we can now prepare pictures for the training dataset
   - preparation of a dataset with pictures of formations from a stock chart according to a certain logic:
     - the closing price and two moving averages are drawn on the picture - the pictures are drawn for the lower timeframe
     - if on the higher timeframe the close is higher than the previous close, then class **1** is assigned to such a picture, otherwise **0**
   - To do this, use the file **2_prepare_dataset_images_from_historical_data.py**
   - The resulting pictures are stored in the **NN\training_dataset_M1** in the subdirectories of classifications **0** and **1**.
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/neural_network_training.png )


4. Finally, there are datasets for training the neural network)) Now we train the neural network 
   - Using a Convolutional Neural Network (CNN)
   - For this, the file **3_train_neural_network.py** is used
   - The neural network training log is located in the file **3_results_of_training_neural_network.txt**
   - The convergence of the neural network is in the file **3_Training and Validation Accuracy and Loss.png**
   - When training a neural network, model files are saved to the **NN\\_models** 
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/training.png )


5. After successfully training the neural network, you need to choose one of the trained models for our trading robot
   - The choice of the best trained model is based on the parameters loss, accuracy, val_loss, val_accuracy
   - The selected model must be **manually** saved to the **NN_winner** under the name **cnn_Open.hdf5**
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/accuracy_loss.png )


6. Now you need to check the predictions made by the neural network on a part of the classified pictures
   - For this, the file **4_check_predictions_by_neural_network.py** is used
   - Just check - that's all OK
   ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/classification.png )


7. Finally, we are checking the connection to the Finam API so that we can trade
   - For this, the file **5_test_api_finam_v1.py** is used - we use [FinamPy](https://github.com/cia76/FinamPy ) for tests
   - and file  **6_test_api_finam_v2.py** use [FinamTradeApiPy](https://github.com/DBoyara/FinamTradeApiPy ) for tests

      ####  How to get Finam API token:
      - Open an account in Finam broker https://open.finam.ru/registration
      - Register in Comon service https://www.comon.ru/
      - In your Comon account, get a token https://www.comon.ru/my/trade-api/tokens for the selected trading account
      - Copy and paste the received **API Key** and **trading account number** into the **my_config\Config.py** file (an example of the config file is here: **my_config\Config_example.py** )
      
      ```python
      # content of my_config\Config.py 
      class Config:
          ClientIds = ('<Trading account>',)  # Trading Accounts
          AccessToken = '<Token>'  # Trade Access Token
      ```


8. Now we are ready to launch the trading robot in live mode
      - Do not forget about the **API Key** and **trading account number** , they should already be registered in the **my_config\\Config.py** file
      - the live strategy is launched using the **7_live_strategy.py** file
      - in line 266 this **days_back** parameter is responsible for how many days back to take the data, if you run the script on Monday or after a weekend/holiday, then increase this value
   
          ```days_back = 1  # how many days ago do we take the data```

      - line 206 can be uncommented so that the script does not run, for example, if the market is not open (weekends and holidays are not taken into account)
   
          ```await self.ensure_market_open()  # проверяем, что рынок открыт```

      - the configuration of the trading robot is located in the file **my_config\\trade_config.py** Since we take historical data from MOEX, and not from the Finam API (because such functionality will be implemented later), 
      - then available tickers for analytics and data download must be selected manually.
          ```
            training_NN = {"SBER", "VTBR"}  # tickers by which we train the neural network
            portfolio = {"SBER", "VTBR"}  # tickers for which we trade and download historical data
           ```
      - the trading robot launch log in live trading mode is located in the file **7_live_trading_log.txt** 
      
      trading robot and trading terminal launched in live mode:
      ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/live_orders_2_r.jpg )
      
      orders placed by the trading robot:
      ![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/live_orders_r.jpg )


Now you can run and watch, but it's better to watch the video on working with this code, posted [on YouTube](https://youtu.be/yrQFqvc4fk0 ) and [on RuTube](https://rutube.ru/video/private/1255dfe65f4db8736b894cae72b14c45/?p=oOFSDPr1El6lq586tIm2qg )

### Attention
Some files contain a line:
```exit(777)  # to prohibit the code from running, otherwise it will rewrite the results```
this is done on purpose so as not to accidentally overwrite the data, it can be commented out when you test your models and your settings.


P.S. The strategy code does not implement a check for the availability of funds on the account to enter a trade.


The code was tested on ```M1=>M10``` and ```M10=>H1```, for other timeframes it is necessary to create a larger number of training samples.


Performance was tested on ```Python 3.10+``` and ```Python 3.11+``` with the latest versions of the libraries.


==========================================================================

## Thanks
- [FinamPy](https://github.com/cia76/FinamPy ): Igor for the library that allows you to work with the functionality of the Finam broker API.
- [FinamTradeApiPy](https://github.com/DBoyara/FinamTradeApiPy ): DBoyara for the library, an asynchronous REST client for the Finam API.
- tensorflow: For a simple and cool library for working with neural networks.
- aiomoex: For the good implementation of getting data from moex.

## Important
Bug fixing, revision and development of the code is carried out by the author and the community!

**Push your commits!** 

# Terms of Use
The program code posted at https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot on the Internet, which allows trading in the stock market using neural networks, is a **Program** created solely for the convenience of working and studying the applications of neural networks /artificial intelligence. 
When using the **Program**, the User is obliged to comply with the provisions of the current legislation of the Russian Federation or his country. 
Use of the **Program** is offered on an "as is" ("AS IS") basis. No warranties, either oral or written, are attached or provided. 
The author and the community do not guarantee that all errors of the **Program** have been eliminated, respectively, the author and the community do not bear any responsibility for the consequences of using the **Program** , including, but not limited to, any damage to the equipment, computers, mobile devices, software of the User caused by or associated with the use of the **Program** , as well as for any financial losses incurred by the User as a result of using the **Program** . No one is liable for data loss, loss, damages, including incidental or consequential, loss of profits, loss of income or any other losses associated with the use of the **Program** .

**The program** is distributed under the terms of the [MIT](https://choosealicense.com/licenses/mit) license.
