#include "ll_func.h"
#include "parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#pragma warning(disable : 4996)

#define CONFUSION 4     //混合行列のサイズ


//評価関数
double Cost_Function(double *y, double *t, int size)
{
    double e = 0.0;

    for(int k = 1; k <= size; k++){
        e -= t[k] * log(y[k]);
    }

    return e;
}


//順伝搬
void forward(LL_PARAM ll_param, double *data, double **w, double **layer_in, double **layer_out)
{
    int i, j, l;     //制御変数
    int h = ll_param.num_unit[0];
    int k = ll_param.output_layer_size;   //クラス数
    int m = ll_param.component_num;    //コンポーネント数
    double exp_sum;   //中間層での出力計算時の分母


    /**************入力層**************/
    layer_in[0] = data;
    layer_out[0] = layer_in[0];

    /**************中間層**************/
    //中間層の入力
    for(i = 1; i <= k*m; i++){
      layer_in[1][i] = 0.0;

      for(j = 1; j <= h; j++){
        layer_in[1][i] += w[j][i] * layer_out[0][j];
      }
    }

    //中間層の出力
    //分母の計算
    exp_sum = 0.0;
    for(i = 1; i <= k*m; i++){
      exp_sum += exp(layer_in[1][i]);
    }

    for(i = 1; i <= k*m; i++){
      layer_out[1][i] = exp(layer_in[1][i]) / exp_sum;
    }

    /**************出力層**************/
    //中間層の素子数分ループし，出力層の入力には中間層の出力のコンポーネントごとの和を入れる
    l = 0;
    for(i = 1; i <= k*m; i++){
        //コンポーネントの区切りごとに出力層の入力を初期化する
        if(i % m == 1){
            l++;
            layer_in[2][l] = 0.0;
        }

        layer_in[2][l] += layer_out[1][i];
    }

    //出力
    for(i = 1; i <= k; i++){
      layer_out[2][i] = layer_in[2][i];
    }
}


//重みの更新
void update_w(LL_PARAM ll_param, double epsilon, double **w, double *t, double **layer_out)
{
    int i, j, l;     //制御変数
    int h = ll_param.num_unit[0];
    int k = ll_param.output_layer_size;   //クラス数
    int m = ll_param.component_num;    //コンポーネント数
    double dJ_dw = 0.0;   //評価関数の微分

    for(i = 1; i <= h; i++){    //i : 入力の次元のインデックス
        for(j = 1; j <= k; j++){    //j : クラスのインデックス
            for(l = 1; l <= m; l++){    //l : コンポーネントのインデックス
                if(((j - 1) * m + l) != (k * m + 1)){
                    //微分値の計算
                    dJ_dw = (layer_out[2][j] - t[j]) * layer_out[1][(j-1)*m + l] * layer_out[0][i] / layer_out[2][j];
                }

                //更新
                w[i][(j-1)*m + l] -= epsilon * dJ_dw;
            }
        }
    }
}


void batch_update_w(LL_PARAM ll_param, double epsilon, double **w, double **t, double ***layer_out, int batch_size)
{
    int i, j, l, n;     //制御変数
    int h = ll_param.num_unit[0];
    int k = ll_param.output_layer_size;   //クラス数
    int m = ll_param.component_num;    //コンポーネント数
    double sum_dJ_dw;   //評価関数の微分の総和
    double dJ_dw;       //評価関数の微分
    //FILE *fp;

    /*
    fp = fopen("dJ_dw_batch.csv", "w");
    if (fp == NULL) {
        printf("ファイルが開けません\n");
        exit(EXIT_FAILURE);
    }
    */


    for (i = 1; i <= h; i++) {    //i : 入力の次元のインデックス
        for (j = 1; j <= k; j++) {    //j : クラスのインデックス
            for (l = 1; l <= m; l++) {    //l : コンポーネントのインデックス
                sum_dJ_dw = 0.0;

                if (((j - 1) * m + l) != (k * m)) {
                    for (n = 0; n < batch_size; n++) {
                        dJ_dw = (layer_out[n][2][j] - t[n][j]) * layer_out[n][1][(j - 1) * m + l] * layer_out[n][0][i] / layer_out[n][2][j];
                        sum_dJ_dw += dJ_dw;
                        //fprintf(fp, "%d,%d,%d,%d,%lf\n", i, j, l, n, dJ_dw);
                    }
                }

                //更新
                w[i][(j - 1) * m + l] -= epsilon * sum_dJ_dw;
            }
        }
    }

    //fclose(fp);

    i = 0;
}


//構造体の設定
LL_PARAM set_param(LL_PARAM ll_param)
{
    //num_unitのメモリ確保
    if((ll_param.num_unit = (int*)malloc((LL_N) * sizeof(int))) == NULL){
        exit(-1);
    }

    //各層の素子数
    ll_param.num_unit[0] = 1 + ll_param.input_layer_size * (ll_param.input_layer_size + 3) / 2;
    ll_param.num_unit[1] = ll_param.output_layer_size * ll_param.component_num;     //K*Mk
    ll_param.num_unit[2] = ll_param.output_layer_size;                              //K

    return ll_param;
}


//入力ベクトルの非線形変換
void Non_linear_tranform(LL_PARAM ll_param, double **input_x, double **output_x, int data_num)
{
    int i,j,k, n;  //制御変数
    int d = ll_param.input_layer_size;  //入力ベクトルの次元

    for (n = 0; n < data_num; n++) {
        //第一項は1
        output_x[n][1] = 1;


        //第二項～は入力ベクトルx
        for(i = 1; i <= d; i++){
            output_x[n][i+1] = input_x[n][i];
        }


        //第三項以降
        k = d;  //入力ベクトルXのインデックス
        for (i = 1; i <= d; i++) {
            j = 0;
            while (i + j <= d) {
                output_x[n][k + 2] = input_x[n][i] * input_x[n][i + j];
                //printf("output_x[%d][1] = input_x[%d] * input_x[%d] = %lf\n", k+3,i,j+i,output_x[k+3][1]);
                j++;
                k++;
            }
        }
    }
}


//ターミナルラーニング
void TA_update_w(LL_PARAM ll_param, double** w, double* t, double** layer_out, double J0, double beta, int tf, double J, double delta_t)
{
    int i, j, l;     //制御変数
    int h = ll_param.num_unit[0];
    int k = ll_param.output_layer_size;   //クラス数
    int m = ll_param.component_num;    //コンポーネント数
    double dJ_dw = 0.0;   //評価関数の微分
    double eta;
    double gamma = 0.0;

    //学習率etaの計算
    eta = pow(J0, 1 - beta) / (tf * (1 - beta));

    for (i = 1; i <= h; i++) {    //i : 入力の次元のインデックス
        for (j = 1; j <= k; j++) {    //j : クラスのインデックス
            for (l = 1; l <= m; l++) {    //l : コンポーネントのインデックス
                if (((j - 1) * m + l) != (k * m)) {
                    //微分値の計算
                    dJ_dw = (layer_out[2][j] - t[j]) * layer_out[1][(j - 1) * m + l] * layer_out[0][i] / layer_out[2][j];

                    //gammaの計算
                    gamma = pow(J, beta) / pow(dJ_dw, 2.0);
                }

                //更新
                w[i][(j - 1) * m + l] -= (delta_t / 2) * eta * gamma * dJ_dw;
            }
        }
    }
}


void TA_batch_update_w(LL_PARAM ll_param, double** w, double** t, double*** layer_out, double J0, double beta, int tf, double delta_t, double *J, int batch_size)
{
    int i, j, l, n;     //制御変数
    int h = ll_param.num_unit[0];
    int k = ll_param.output_layer_size;   //クラス数
    int m = ll_param.component_num;    //コンポーネント数
    double dJ_dw = 0.0;   //評価関数の微分
    double sum_dJ_dw;
    double eta;
    double gamma = 0.0;
    double sum_J;

    //学習率etaの計算
    eta = pow(J0, 1 - beta) / (tf * (1 - beta));

    for (i = 1; i <= h; i++) {    //i : 入力の次元のインデックス
        for (j = 1; j <= k; j++) {    //j : クラスのインデックス
            for (l = 1; l <= m; l++) {    //l : コンポーネントのインデックス
                sum_dJ_dw = 0.0;
                sum_J = 0.0;
                gamma = 0.0;

                if (((j - 1) * m + l) != (k * m)) {
                    for (n = 0; n < batch_size; n++) {
                        //微分値の計算
                        dJ_dw = (layer_out[n][2][j] - t[n][j]) * layer_out[n][1][(j - 1) * m + l] * layer_out[n][0][i] / layer_out[n][2][j];
                        sum_dJ_dw += dJ_dw;
                        sum_J += J[n];
                    }


                    //gammaの計算
                    gamma = pow(sum_J, beta) / pow(sum_dJ_dw, 2.0);
                }

                //更新
                w[i][(j - 1) * m + l] -= (delta_t / 2 ) * eta * gamma * dJ_dw;
            }
        }
    }
}


//正解率の計算
double Accuracy(LL_PARAM ll_param, double*** layer_out, double** t, int data_num)
{
    int i, j;
    double correct = 0.0;
    double correct_rate;

    for (i = 0; i < data_num; i++) {
        for (j = 1; j <= ll_param.output_layer_size; j++) {
            if (fabs(layer_out[i][2][j] - t[i][j]) <= 0.1) {
                correct++;
            }
        }
    }

    correct_rate = correct / (data_num * ll_param.output_layer_size);

    return correct_rate;
}


//識別率
double Identification(LL_PARAM ll_param, double*** layer_out, double** t, int data_num)
{
    int i, j;
    double identify_rate = 0.0;
    int max_rate_num;
    int correct_num;

    for (i = 0; i < data_num; i++) {
        max_rate_num = 1;
        correct_num = 1;
        for (j = 2; j <= ll_param.output_layer_size; j++) {
            if (layer_out[i][2][max_rate_num] < layer_out[i][2][j]) {
                max_rate_num = j;
            }

            if (t[i][j] == 1) {
                correct_num = j;
            }
        }

        if (max_rate_num == correct_num) {
            identify_rate++;
        }
    }

    identify_rate = identify_rate / data_num;

    return identify_rate;
}
