#include "ll_func.h"
#include "parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


//評価関数
double Cost_Function(double *y, double *t, int size)
{
    double e = 0.0; //sum((y-t)^2)

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

    for(i = 1; i<= k*m; i++){
      layer_out[1][i] = exp(layer_in[1][0]) / exp_sum;
    }

    /**************出力層**************/
    //中間層の素子数分ループし，出力層の入力には中間層の出力のコンポーネントごとの和を入れる
    l = 0;
    for(i = 1; i <= k*m; i++){
        //コンポーネントの区切りごとに出力層の入力を初期化する
        if(i % k == 0){
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
                if(j*l < k*m){
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
    double sum_dJ_dw;   //評価関数の微分

    for(n = 1; n <= batch_size; n++){
        for(i = 1; i <= h; i++){    //i : 入力の次元のインデックス
            for(j = 1; j <= k; j++){    //j : クラスのインデックス
                for(l = 1; l <= m; l++){    //l : コンポーネントのインデックス
                    sum_dJ_dw = 0.0;

                    if(j*l < k*m){
                        sum_dJ_dw += (layer_out[n][2][j] - t[n][j]) * layer_out[n][1][(j-1)*m + l] * layer_out[n][0][i] / layer_out[n][2][j];

                    }

                    //更新
                    w[i][(j-1)*m + l] -= epsilon * sum_dJ_dw;
                }
            }
        }
    }
}

//構造体の設定
LL_PARAM set_param(LL_PARAM ll_param)
{
    //num_unitのメモリ確保
    if((ll_param.num_unit = (int*)malloc((LL_N) * sizeof(int))) == NULL){
        exit(-1);
    }

    //各層の素子数
    ll_param.num_unit[0] = 1;
    for(int i = 1; i <= ll_param.input_layer_size; i++){
      ll_param.num_unit[0] += i;
    }
    ll_param.num_unit[1] = ll_param.output_layer_size * ll_param.component_num;     //K*Mk
    ll_param.num_unit[2] = ll_param.output_layer_size;                              //K

    return ll_param;
}


//入力ベクトルの非線形変換
void Non_linear_tranform(LL_PARAM ll_param, double **input_x, double **output_x)
{
    int i,j,k, n;  //制御変数
    int d = ll_param.input_layer_size;  //入力ベクトルの次元

    for (n = 0; n < DATA_N; n++) {
        //第一項は1
        output_x[n][1] = 1;

        /*
        //第二項は入力ベクトルx
        for(i = 1; i <= d; i++){
            output_x[2][i] = input_x[i];
        }
        */

        //第三項以降
        k = 0;  //入力ベクトルXのインデックス
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
