uses MLABC;

begin
  // Установка языка для вывода системных сообщений
  Datasets.Language := 'ru';
  
  // Загрузка локального датасета
  var df := DataFrame.FromCsv('../data/handwriting_minimal_dataset.csv');
  
  // Вывод первых 20 строк данных
  Println('Первые 10 строк датасета:');
  df.Head(10).Print;
end.