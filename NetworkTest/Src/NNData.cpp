#include "NNData.h"

//--------------------------------------------
// Читает значения иконок из файла
//
size_t NNData::Read(const std::string& file_name) {

    std::ifstream data_file;
    data_file.open(file_name);
    if (!data_file.is_open()) { return -1; }
    //  Файл открылся начинаем читать
    size_t row_count = 0;
    std::string token, line_read;

    while (!data_file.eof()) { // Пока не дойдем до конца файла
        data_file >> line_read;
        if (line_read.size() == 0) { return -1; }
        icon_vect_.push_back(NNIcon()); //Добавляем очередную иконку в вектор иконок
        std::stringstream line_stream(line_read); //Делаем из строки поток для последовательного считывания значений

        int token_count = 0;
        while (getline(line_stream, token, ',')) {  // Считываем очередное значение до запятой в строку token
            try {
                if (token_count == NN_ICON_SIZE) //Если значение равно 64, значит токен содержит цифровое значение иконки
                    icon_vect_[row_count].value_ = stoi(token);
                else
                    icon_vect_[row_count].data_[token_count] = stoi(token); //Заполняем массив значениями пикселей иконки

            }
            // Обрабатываем исключения
            catch (std::invalid_argument const& ex) {

                return -1;
            }
            catch (std::out_of_range const& ex) {

                return -1;
            }

            // Проверка на превышение количества токенов
            if (token_count > NN_ICON_SIZE) { return -1; }
            token_count++;
        }
        //Проверим, что количество считанных значений соответствует расчетному
        if (token_count < NN_ICON_SIZE + 1) { return -1; }

        row_count++;
    }

    data_file.close();
    return row_count;
}

//-----------------------------------------------
// Подготавливает данные иконок с помощью выбранного в битовом поле prep_way метода
//
void NNData::Prepare(unsigned prep_way)
{

    if (prepared_) return; //Данные уже подготовлены
    float min = 0.0F, max = 0.0F;
    float wa_icon = 0.0F;
    float wa = 0.0F, fmax = 0.0F;
    if (icon_vect_.size() == 0)
    {
        //std::cout << "Данных для подготовки нет \n";
        return;
    }
    min = max = icon_vect_[0].data_[0];

    for (auto& icon : icon_vect_)
    {
        wa_icon = 0;
        for (int i = 0; i < NN_ICON_SIZE; i++)
        {
            min = icon.data_[i] < min ? icon.data_[i] : min; //Находим минимум и максимум значений в иконке
            max = icon.data_[i] > max ? icon.data_[i] : max;
            wa_icon += icon.data_[i]; //Для взвешенного суммируем все значения иконки
        }
        wa_icon = wa_icon / (float)(NN_ICON_SIZE); //Средневзвешенное значение данных иконки
        wa += wa_icon / icon_vect_.size(); // Прибавляем к средневзвешенному по всей выборке взвешивая по количеству иконок
    }
    wa = wa - min; // Опускаем средневзвешенное

    if ((prep_way & (unsigned)DATA_PREP_UNSIG) == (unsigned)DATA_PREP_UNSIG) { //Значения данных будут от 0 до 1
        for (auto& icon : icon_vect_) {
            for (int i = 0; i < NN_ICON_SIZE; i++) {
                icon.data_[i] = (icon.data_[i] - min) / (max - min);
            }
        }
    }
    else if ((prep_way & (unsigned)DATA_PREP_ZERO_AV) == (unsigned)DATA_PREP_ZERO_AV) { // Значения от -1 до 1 со средним арифметическим в нуле
        for (auto& icon : icon_vect_) {
            for (int i = 0; i < NN_ICON_SIZE; i++) {
                fmax = max - min;
                icon.data_[i] = ((icon.data_[i] - min) - fmax / 2.0F) / (fmax / 2.0F);
            }
        }
    }
    else if ((prep_way & (unsigned)DATA_PREP_ZERO_WAV) == (unsigned)DATA_PREP_ZERO_WAV) { // Значения от -1 до 1 со средневзвешенным в нуле
        float f_divider = (max - wa) > (wa - min) ? max - wa : wa - min;
        for (auto& icon : icon_vect_) {
            for (int i = 0; i < NN_ICON_SIZE; i++) {

                icon.data_[i] = ((icon.data_[i] - min) - wa) / f_divider;
            }
        }
    }
}
