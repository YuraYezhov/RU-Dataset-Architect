uses MLABC;

begin
  // Установка языка для вывода системных сообщений
  Datasets.Language := 'ru';
  
  // Загрузка локального датасета
  var df := DataFrame.FromCsv('../data/domes_medium_dataset_easy.csv');
  
  // Вывод первых 10 строк данных
  Println('Первые 10 строк датасета:');
  df.Head(10).Print;
end.