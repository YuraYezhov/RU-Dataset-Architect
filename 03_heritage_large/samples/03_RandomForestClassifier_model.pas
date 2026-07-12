uses MLABC;

begin
  // Установка языка для вывода системных сообщений
  Datasets.Language := 'ru';
  
  // Загрузка локального датасета
  var df := DataFrame.FromCsv('../data/culture_large_dataset_easy.csv');
  
  // Создаем массив названий
  // 0 - ART, 1 - DOCUMENTS, 2 - ICON, 3 - NUMISMATICS
  var myNames := ['ART', 'DOCUMENTS', 'ICON', 'NUMISMATICS'];
  
  // Подготовка признаков (X)
  var featureNames := ['height', 'width', 'length', 'weight', 
                       'is_painting_hint', 'is_icon_hint', 'is_coin_hint', 'is_doc_hint'];
  var X := df.ToMatrix(featureNames);
  
  // Подготовка целевой переменной (y)
  var target := df.EncodeTarget('target_label');
  
  // Разделение данных (80% на обучение, 20% на тест)
  // testRatio := 0.2 означает, что 20% данных уйдут в тест
  // seed := 42 нужен, чтобы разделение было всегда одинаковым при перезапуске
  var (X_train, X_test, y_train, y_test) := Validation.TrainTestSplit(X, target.Labels, testRatio := 0.2, seed := 42);
  
  // Создание и обучение модели RandomForestClassifier
  // Параметры nTrees (кол-во деревьев) и maxDepth взяты из примера на фото 2
  var model := new RandomForestClassifier(nTrees := 20, maxDepth := 20, seed := 42);
  model.Fit(X_train, y_train);
  
  // Оценка результатов на тестовой выборке
  var predictions := model.Predict(X_test);
  var report := Metrics.ClassificationReport(y_test, predictions);
  
  Println('--- Отчет о качестве (Random Forest) ---');
  Println(report);
  Println;
  
  // Вывод точности
  var acc := Metrics.Accuracy(y_test, predictions);
  Println('Точность:', acc:0:3);
  Println;
  
  // Демонстрация предсказания
  var cnt := 10; // Сколько примеров вывести
  Println($'--- Примеры прогнозов (первые {cnt}) ---');
  
  for var i := 0 to cnt - 1 do
  begin
    // Получаем названия классов по их числовым индексам
    var realName := myNames[y_test[i]];
    var predName := myNames[predictions[i]];
    
    // Ровный вывод таблицы через фиксированную ширину (:11)
    Println('Реально: ', realName:11, ' | Предсказано: ', predName);
  end;
end.