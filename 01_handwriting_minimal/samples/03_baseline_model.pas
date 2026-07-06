uses MLABC;

begin
  // Установка языка для вывода системных сообщений
  Datasets.Language := 'ru';
  
  // Загрузка локального датасета
  var df := DataFrame.FromCsv('../data/handwriting_minimal_dataset.csv');  
  
  // Подготовка признаков (X)
  // Выбираем все 5 числовых параметров
  var X := df.ToMatrix(['aspect_ratio', 'ink_density', 'components', 'peaks', 'solidity']);
  
  // Подготовка целевой переменной (y)
  var y := df.ToVector('target_letters');
  
  // Разделение данных (80% на обучение, 20% на тест)
  // testRatio := 0.2 означает, что 20% данных уйдут в тест
  // seed := 42 нужен, чтобы разделение было всегда одинаковым при перезапуске
  var (X_train, X_test, y_train, y_test) := Validation.TrainTestSplit(X, y, testRatio := 0.2, seed := 42);
  
  // Создание и обучение модели - линейная регрессия
  var model := new LinearRegression();
  model.Fit(X_train, y_train);
  
  // Проверка на тестовой выборке
  var testPreds := model.Predict(X_test);
  
  // Вывод результатов для первых 5 объектов из тестовой выборки
  Println('--- Сравнение результатов на тестовых данных (20%) ---');
  Println('№ | Реально (букв) | Предсказано | Ошибка');
  Println('-' * 50);
  
  for var i := 0 to 4 do
  begin
    var realVal := y_test[i];
    var predVal := testPreds[i];
    var error := Abs(realVal - predVal);
    
    Println(i + 1, ' | ', realVal:14:0, ' | ', predVal:11:2, ' | ', error:6:2);
  end;
  
end.