#include <stdlib.h>
#ifndef __INC_PARAMETERS_H
#define __INC_PARAMETERS_H

#define N 1000        //試行回数の最大値
#define LOSS_MIN 1  //損失の閾値
#define LL_N 3          //LLGMNの層数
#define DATA_N 800      //データ数

//構造体
typedef struct {
    int input_layer_size;   //入力層のサイズ
    int component_num;      //中間層のサイズ
    int output_layer_size;  //出力層のサイズ
    int *num_unit;              //各層の素子数
} LL_PARAM;

//変数
double **train_data;        //入力データ
double **w;                //重み
double ***layer_in;          //各層の入力
double ***layer_out;         //各層の出力
double **t;                  //正解データ
double **unlearn_data;      //未学習データ
double **output_x;          //非線形変換後の入力ベクトル
double *J;                  //評価関数
double **dis_t;               //未学習データの正解データ
double ***dis_layer_in;       //未学習データの各層の入力
double ***dis_layer_out;      //未学習データの各層の出力
double **dis_output_x;               //未学習データの非線形変換後の入力ベクトル

#endif
