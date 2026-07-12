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

  // Построение и обучение модели KNNClassifier
  // Используем 5 ближайших соседей
  var model := new KNNClassifier(5);
  model.Fit(X_train, y_train);

  // Оценка результатов на тестовой выборке
  var predictions := model.Predict(X_test);
  var report := Metrics.ClassificationReport(y_test, predictions);

  Println('--- Отчет о качестве (KNN Classifier) ---');
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
    var realName := target.ClassNames[y_test[i]];
    var predName := target.ClassNames[predictions[i]];

    Println('Реально: ', realName:12, ' | Предсказано: ', predName);
  end;

  // Анализ соседей для одного примера
  Println;
  Println('--- Анализ ближайших соседей для первого тестового объекта ---');
  var example := X_test.Row(0);
  var neigh := model.GetNearestNeighbors(example);
  
  Println('Ближайшие соседи (Класс и Дистанция):');
  foreach var n in neigh do
  begin
    // Узнаем ID класса (0, 1, 2 или 3) для соседа по его индексу в y_train
    var classId := Trunc(y_train[n.Index]); 
    
    // Используем этот ID, чтобы достать имя из встроенного списка target.ClassNames
    Println(target.ClassNames[classId]:12, n.Distance:0:2);
  end;
end.