uses MLABC;

begin
   // Установка языка для вывода системных сообщений
  Datasets.Language := 'ru';
  
  // Загрузка локального датасета
  var df := DataFrame.FromCsv('../data/domes_medium_dataset_easy.csv');
  
  // Подготовка признаков (X)
  var featureNames := ['aspect_ratio', 'solidity', 'extent', 'top_width_ratio', 'hu_moment_1', 'century'];
  var X := df.ToMatrix(featureNames);
  
  // Подготовка целевой переменной (y)
  var target := df.EncodeTarget('target_shape');

  // Разделение данных (80% на обучение, 20% на тест)
  var (X_train, X_test, y_train, y_test) := Validation.TrainTestSplit(X, target.Labels, 0.2, 42);

  // Построение и обучение модели RandomForestClassifier
  var model := new RandomForestClassifier(nTrees := 100, maxDepth := 20, seed := 42);
  model.Fit(X_train, y_train);

  // Оценка результатов на тестовой выборке
  var predictions := model.Predict(X_test);
  var report := Metrics.ClassificationReport(y_test, predictions);
  
  Println('--- Отчет о качестве (Random Forest Balanced) ---');
  Println(report);
  Println;
  
  // Вывод точности
  var acc := Metrics.Accuracy(y_test, predictions);
  Println('Точность:', acc:0:3);
  Println;
  
  // Демонстрация предсказания
  var cnt := 10;
  Println($'--- Примеры прогнозов (первые {cnt}) ---');
  
  for var i := 0 to cnt - 1 do
  begin
    var realName := target.ClassNames[y_test[i]];
    var predName := target.ClassNames[predictions[i]];
    
    // Форматированный вывод
    Println('Реально: ', realName:11, ' | Предсказано: ', predName);
  end;
end.