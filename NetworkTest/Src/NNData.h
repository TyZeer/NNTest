#include <vector>
#include <iostream>
#include "fstream"
#pragma omce

#include "string"
#include "sstream"

#define NN_ICON_SIZE 8*8

#define DATA_PREP_UNSIG 1       // (val-min)/(max-min)
#define DATA_PREP_ZERO_AV 2     // (val-min - (max-min)/2) / (max-min)/2
#define DATA_PREP_ZERO_WAV 4    // (val-min - (WA-min) / (WA-min)

// Структура данных иконки с нарисованной цифрой
struct NNIcon {
    float data_[NN_ICON_SIZE];  //Данные значений пикселов иконки из файла НУЖНО ОТ ЭТОГО УЙТИ
    unsigned value_;     //Цифра, нарисованная в иконке
};

// Класс, содержащий данные всех загруженных иконок
class NNData {
private:
    bool prepared_ = false; //Подготовлены ли данные иконок
public:
    std::vector<NNIcon> icon_vect_; //Вектор с иконками

    size_t Read(const std::string& file_name); //Считывает данные иконок из файла
    void Prepare(unsigned prep_way = DATA_PREP_ZERO_WAV); //Предварительная обработка данных файла
};


