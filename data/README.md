# Данные температур

Используется ряд **Daily minimum temperatures in Melbourne, Australia, 1981–1990**. [Описание источника](https://github.com/jbrownlee/Datasets/blob/d20fcb6402ae34e653d4513b00f39257bb37ed7f/daily-min-temperatures.names) ссылается на Time Series Data Library и Australian Bureau of Meteorology.

Зеркало CSV: [daily-min-temperatures.csv](https://github.com/jbrownlee/Datasets/blob/d20fcb6402ae34e653d4513b00f39257bb37ed7f/daily-min-temperatures.csv). Для проекта файл нужно сохранить как `temperatures.csv` рядом с этим README. Формат — UTF-8 CSV со столбцами `Date` и `Temp`, температура в °C.

При проверке все 3650 записей локального файла точно совпали с зеркалом после чтения CSV. Разница исходных файлов — завершающие пробельные символы. Пропущенных значений и повторных дат нет; между соседними датами найдены два интервала, отличающихся от суток. Скрипты не заполняют отсутствующие даты.

Корпус и веса в репозитории проекта не распространяются. Лицензия на пользовательский код не подменяет условия исходных данных. Команда получения CSV приведена в основном README.
