uses MLABC;

begin
  // Установка языка для вывода системных сообщений
  Datasets.Language := 'ru';
  
  // Загрузка локального датасета
  var df := DataFrame.FromCsv('../data/handwriting_minimal_dataset.csv');
  
  // Вывод схемы данных: названия столбцов, количество непустых значений и их типы
  Println('Схема данных:');
  df.PrintInfo;
  Println;
  
  // Вывод статистических показателей (среднее, минимум, максимум, квантили)
  Println('Базовая статистика по всем признакам:');
  df.Describe.Print;
end.