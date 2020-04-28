#include <stdio.h>
#include "ll_func.h"
#include "parameters.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <conio.h>
#pragma warning(disable : 4996)

#define FILENAME 30     //ファイルの最大文字数

int main(void){
    int i, j, k;        //制御変数
    int key;  //対話用選択肢
    LL_PARAM ll_param;  //構造体
    FILE *fp;
    char filename[FILENAME];    //ファイル名
    double Loss_batch;
    double Loss_seq;        //損失関数
    double epsilon;     //学習率
    int batch_count;    //一括学習回数
    int seq_count;      //逐次学習回数
    double beta;        //TAの定数
    int tf;             //TAの学習回数
    double sampling_time; //サンプリング時間
    double J0;          //評価関数の初期値
    int n_random;                   //入力ベクトルのシャッフルに使用
    double* tmp = NULL;             //入力ベクトルのシャッフルに使用
    double* tmp_t = NULL;           //正解データのシャッフルに使用
    double correct_rate;            //正解率

    int train_num;                  //教師データのデータ数
    int dis_num;                    //未学習データのデータ数

    int e_count = 0;                //key 'e' を選択した回数


    /**************************層数・素子数の設定*****************************/

    printf("コンポーネント数を入力してください：\n");
    scanf("%d", &ll_param.component_num);

    printf("入力次元数を入力してください：\n");
    scanf("%d", &ll_param.input_layer_size);

    printf("クラス数を入力してください：\n");
    scanf("%d", &ll_param.output_layer_size);

    printf("教師データのファイル名を入力してください：\n");
    scanf("%s", filename);

    printf("教師データのデータ数を入力してください：\n");
    scanf("%d", &train_num);



    //ll_param.component_num = 2;
    //ll_param.input_layer_size = 2;
    //ll_param.output_layer_size = 4;

    /**************************各種パラメータの設定*****************************/
    ll_param = set_param(ll_param);

    double **train_data = NULL;     //入力データ
    double **w = NULL;              //重み
    double ***layer_in = NULL;       //各層の入力
    double ***layer_out = NULL;      //各層の出力
    double **t = NULL;              //正解データ
    double **unlearn_data;          //未学習データ
    double **output_x = NULL;       //非線形変換後の入力ベクトル
    double *J = NULL;
    double** dis_t = NULL;               //未学習データの正解データ
    double*** dis_layer_in = NULL;       //未学習データの各層の入力
    double*** dis_layer_out = NULL;      //未学習データの各層の出力
    double** dis_output_x = NULL;        //未学習データの非線形変換後の入力ベクトル


    //train_data
    if((train_data = (double**)malloc((train_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= train_num; i++){
        if((train_data[i] = (double*)malloc((ll_param.input_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    //w
    if((w = (double**)malloc((ll_param.num_unit[0] + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    //乱数の初期化
    srand((unsigned int)time(NULL));

    for(i = 0; i <= ll_param.num_unit[0]; i++) {
        if((w[i] = (double*)malloc((ll_param.num_unit[1] + 1) * sizeof(double))) == NULL) {
            exit(-1);
        }

        for(j = 0; j <= ll_param.num_unit[1]; j++) {
            if (j == ll_param.num_unit[1]) {
                w[i][j] = 0.0;
            }
            else {
                w[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  //乱数でwを初期化
            }
        }
    }

    //学習前のパラメータを出力するファイルを開く
    fp = fopen("w.csv", "w");
    if( fp == NULL ){
        printf( "ファイルが開けません\n");
        return -1;
    }

    printf("重み\n");

    for(i = 1; i <= ll_param.num_unit[0]; i++) {
        printf("w[%d] ", i);
        for(j = 1; j <= ll_param.num_unit[1]; j++) {
            printf(" %9lf", w[i][j]);
            fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
        }printf("\n");
    }
    printf("\n");

    fclose(fp);

    //layer_out
    if((layer_out = (double***)malloc((train_num + 1) * sizeof(double**))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= train_num; i++) {
        if((layer_out[i] = (double**)malloc((LL_N) * sizeof(double*))) == NULL) {
            exit(-1);
        }

        for(j = 0; j < LL_N; j++) {
            if((layer_out[i][j] = (double*)malloc((ll_param.num_unit[j] + 1) * sizeof(double))) == NULL) {
                exit(-1);
            }

            for(k = 0; k <= ll_param.num_unit[j]; k++){
                layer_out[i][j][k] = 0.0;
            }
        }
    }

    //layer_in
    if((layer_in = (double***)malloc((train_num + 1) * sizeof(double**))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= train_num; i++) {
        if((layer_in[i] = (double**)malloc((LL_N) * sizeof(double*))) == NULL) {
            exit(-1);
        }

        for(j = 0; j < LL_N; j++) {
            if((layer_in[i][j] = (double*)malloc((ll_param.num_unit[j] + 1) * sizeof(double))) == NULL) {
                exit(-1);
            }

            for(k = 0; k <= ll_param.num_unit[j]; k++){
                layer_in[i][j][k] = 0.0;
            }
        }
    }

    //t
    if((t = (double**)malloc((train_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= train_num; i++){
        if((t[i] = (double*)malloc((ll_param.output_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    for(i = 0; i <= train_num; i++){
        for(j = 0; j <= ll_param.output_layer_size; j++){
            t[i][j] = 0.0;
        }
    }

    //output_x
    if((output_x = (double**)malloc((train_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= train_num; i++){
        if((output_x[i] = (double*)malloc((ll_param.num_unit[0] + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    //J
    if ((J = (double*)malloc((train_num + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }


    /**************************教師データ・正解データの読み込み*****************************/

    //教師データ
    //ファイルオープン
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("ファイルを開けませんでした．\n");
        return -1;
    }
    //データ読み込み
    //printf("教師データ\n");
    i = 0;
    while ((fscanf(fp, "%lf,%lf,%lf,%lf", &train_data[i][1], &train_data[i][2], &train_data[i][3], &train_data[i][4])) != EOF) {
      //printf("%.1lf %.1lf\n", train_data[i][1], train_data[i][2]);
      i++;
    }
    //printf("\n");
    fclose(fp);

    printf("正解データのファイル名を入力してください：\n");
    scanf("%s", filename);

    //正解データ
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("ファイルを開けませんでした．\n");
        return -1;
    }
    //データ読み込み
    //printf("正解データ\n");
    i = 0;
    while ((fscanf(fp, "%lf,%lf,%lf,%lf", &t[i][1], &t[i][2], &t[i][3], &t[i][4])) != EOF) {
      //printf("%.1lf %.1lf %.1lf %.1lf\n", t[i][1], t[i][2], t[i][3], t[i][4]);
      i++;
    }
    //printf("\n");
    fclose(fp);

    /*
    Non_linear_tranform(ll_param, train_data, output_x);

    for (int l = 0; l < train_num; l++) {
        printf("%lf %lf %lf %lf %lf %lf\n", output_x[l][1], output_x[l][2], output_x[l][3], output_x[l][4], output_x[l][5], output_x[l][6]);
    }
    printf("\n");
    */

    /**************************システム*****************************/


    printf("**************************************************\n");
    printf("LLGMN\n");
    printf("**************************************************\n");


    while(1){

      printf("[a] 一括更新学習法\n");
      printf("[b] 逐次更新学習法\n");
      printf("[c] TAを用いた一括更新学習法\n");
      printf("[d] TAを用いた逐次更新学習法\n");
      printf("[e] 学習済みニューロンのテスト\n");
      printf("[f] パラメータのリセット\n");
      printf("[ESC] プログラム終了\n");

      printf("**************************************************\n");
      printf("キーを入力して機能を選択してください：\n");
      key = getch();
      printf("**************************************************\n");

      switch (key) {

        /***************************************************************************************************************************************/
        case 'a':
        /**************************一括学習*************************************/
        printf("学習率を入力してください：\n");
        scanf("%lf", &epsilon);
        printf("**************************************************\n");

        //教師データの非線形変換
        Non_linear_tranform(ll_param, train_data, output_x, train_num);

        printf("一括学習の処理を始めます．\n");
        //損失を格納するファイルを開く
        fp = fopen("loss_batch.csv", "w");
        if( fp == NULL ){
            printf( "ファイルが開けません\n");
            return -1;
        }
        batch_count = 0;    //カウントの初期化
        do{
            //教師データについて一つずつ順伝搬・逆伝搬を行い，重みの更新はエポックごとに行う
            Loss_batch = 0.0;
            for(i = 0; i < train_num; i++){
                //順伝搬
                forward(ll_param, output_x[i], w, layer_in[i], layer_out[i]);
                Loss_batch += Cost_Function(layer_out[i][2], t[i], ll_param.output_layer_size);

                /*
                if (i % 10 == 0) {
                    printf("i = %d : %lf\n", i, Loss_batch);
                }
                */
            }

            //重みの更新
            batch_update_w(ll_param, epsilon, w, t, layer_out, train_num);

            /*
            //更新後のパラメータを出力するファイルを開く
            fp = fopen("w_test.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }

            for (i = 0; i <= ll_param.num_unit[0]; i++) {
                for (j = 0; j <= ll_param.num_unit[1]; j++) {
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }
            }

            fclose(fp);

            //layer_inを出力するファイルを開く
            fp = fopen("in_test.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }

            for (i = 0; i < train_num; i++) {
                fprintf(fp, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, layer_in[i][0][1], layer_in[i][0][2], layer_in[i][0][3], layer_in[i][0][4], layer_in[i][0][5], layer_in[i][0][6], layer_in[i][1][1], layer_in[i][1][2], layer_in[i][1][3], layer_in[i][1][4], layer_in[i][1][5], layer_in[i][1][6], layer_in[i][1][7], layer_in[i][1][8], layer_in[i][2][1], layer_in[i][2][2], layer_in[i][2][3], layer_in[i][2][4]);
            }

            fclose(fp);

            //layer_outを出力するファイルを開く
            fp = fopen("out_test.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }

            for (i = 0; i < train_num; i++) {
                fprintf(fp, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, layer_out[i][0][1], layer_out[i][0][2], layer_out[i][0][3], layer_out[i][0][4], layer_out[i][0][5], layer_out[i][0][6], layer_out[i][1][1], layer_out[i][1][2], layer_out[i][1][3], layer_out[i][1][4], layer_out[i][1][5], layer_out[i][1][6], layer_out[i][1][7], layer_out[i][1][8], layer_out[i][2][1], layer_out[i][2][2], layer_out[i][2][3], layer_out[i][2][4]);
            }

            fclose(fp);
            */


            //カウント
            batch_count++;

            if(batch_count % 1 == 0){
                printf("batch_count = %d : %lf\n", batch_count, Loss_batch);
            }
            fprintf(fp, "%d,%lf\n", batch_count, Loss_batch);
        }while(fabs(Loss_batch) > LOSS_MIN && batch_count < N);

        fclose(fp);

        //学習後のパラメータを出力するファイルを開く
        fp = fopen("w_batch.csv", "w");
        if( fp == NULL ){
            printf( "ファイルが開けません\n");
            return -1;
        }

        for(i = 0; i <= ll_param.num_unit[0]; i++) {
            for(j = 0; j <= ll_param.num_unit[1]; j++) {
                fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
            }
        }

        fclose(fp);


          if(fabs(Loss_batch) < LOSS_MIN){
            printf("損失の大きさが一定値を下回りました。\n");
            printf("損失の大きさ : %lf\n", Loss_batch);
            printf("繰り返し回数は%d\n", batch_count);
        }else if(batch_count >= N){
            printf("繰り返し回数が一定数を超えました。\n");
            printf("損失の大きさ : %lf\n", Loss_batch);
            printf("繰り返し回数は%d\n", batch_count);
          }else{
            printf("損失の大きさ : %lf\n", Loss_batch);
            printf("繰り返し回数は%d\n", batch_count);
          }
          printf("**************************************************\n");
          break;


        /***************************************************************************************************************************************/
        case 'b':
        /**************************逐次学習*************************************/
        printf("学習率を入力してください：\n");
        scanf("%lf", &epsilon);
        printf("**************************************************\n");

        //教師データの非線形変換
        Non_linear_tranform(ll_param, train_data, output_x, train_num);

        srand((unsigned int)time(NULL));

        //教師データをシャッフルする
        for (i = 0; i < train_num; i++) {
            n_random = rand() % train_num;
            tmp = output_x[i];
            tmp_t = t[i];
            output_x[i] = output_x[n_random];
            t[i] = t[n_random];
            output_x[n_random] = tmp;
            t[n_random] = tmp_t;
        }

        /*
        for (int l = 0; l < train_num; l++) {
            printf("%9lf %9lf %9lf %9lf %9lf %9lf\n", output_x[l][1], output_x[l][2], output_x[l][3], output_x[l][4], output_x[l][5], output_x[l][6]);
        }
        printf("\n");
        */

        printf("逐次学習の処理を始めます．\n");
        //損失関数を出力するファイルを開く
        fp = fopen("loss_seq.csv", "w");
        if( fp == NULL ){
            printf( "ファイルが開けません\n");
            return -1;
        }
        seq_count = 0;  //カウントの初期化
        do{
            Loss_seq = 0.0;

            //教師データについて一つずつ順伝搬・逆伝搬・重みの更新を行う
            for(i = 0; i < train_num; i++){
                //順伝搬
                forward(ll_param, output_x[i], w, layer_in[i], layer_out[i]);
                Loss_seq += Cost_Function(layer_out[i][2], t[i], ll_param.output_layer_size);

                /*
                if (i % 10 == 0) {
                    printf("i = %d : %lf\n", i, Loss_seq);
                }
                */

                //重みの更新
                update_w(ll_param, epsilon, w, t[i], layer_out[i]);
            }
            //カウント
            seq_count++;

            if(seq_count % 1 == 0){
                printf("seq_count = %d : %lf\n", seq_count, Loss_seq);
            }

            fprintf(fp, "%d,%lf\n", seq_count, Loss_seq);
        }while((fabs(Loss_seq) > LOSS_MIN) && (seq_count < N));

        fclose(fp);

        //学習後のパラメータを出力するファイルを開く
        fp = fopen("w_seq.csv", "w");
        if( fp == NULL ){
            printf( "ファイルが開けません\n");
            return -1;
        }

        for(i = 0; i <= ll_param.num_unit[0]; i++) {
            for(j = 0; j <= ll_param.num_unit[1]; j++) {
                fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
            }
        }

        fclose(fp);

          if(fabs(Loss_seq) < LOSS_MIN){
            printf("損失の大きさが一定値を下回りました。\n");
            printf("損失の大きさ : %lf\n", Loss_seq);
            printf("繰り返し回数は%d\n", seq_count);
        }else if(seq_count > N){
            printf("繰り返し回数が一定数を超えました。\n");
            printf("損失の大きさ : %lf\n", Loss_seq);
            printf("繰り返し回数は%d\n", seq_count);
          }else{
            printf("損失の大きさ : %lf\n", Loss_seq);
            printf("繰り返し回数は%d\n", seq_count);
          }
          printf("**************************************************\n");
          break;



        /***************************************************************************************************************************************/
        case 'c':
            /**************************TAの一括学習*************************************/
            printf("TAの定数beta（０～１）を入力してください：\n");
            scanf("%lf", &beta);

            printf("TAの学習回数tfを入力してください：\n");
            scanf("%d", &tf);

            printf("TAのサンプリング時間を入力してください：\n");
            scanf("%lf", &sampling_time);
            printf("**************************************************\n");

            //教師データの非線形変換
            Non_linear_tranform(ll_param, train_data, output_x, train_num);

            printf("TAを用いた一括学習の処理を始めます．\n");
            //損失を格納するファイルを開く
            fp = fopen("loss_batch.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }
            batch_count = 0;    //カウントの初期化
            do {
                //教師データについて一つずつ順伝搬・逆伝搬を行い，重みの更新はエポックごとに行う
                Loss_batch = 0.0;
                for (i = 0; i < train_num; i++) {
                    //順伝搬
                    forward(ll_param, output_x[i], w, layer_in[i], layer_out[i]);
                    J[i] = Cost_Function(layer_out[i][2], t[i], ll_param.output_layer_size);
                    Loss_batch += J[i];

                    /*
                    if (i % 10 == 0) {
                        printf("i = %d : %lf\n", i, Loss_batch);
                    }
                    */
                }

                if (batch_count == 0) {
                    J0 = Loss_batch;
                }

                //重みの更新
                TA_batch_update_w(ll_param, w, t, layer_out, J0, beta, tf, sampling_time, J, train_num);

                //カウント
                batch_count++;

                if (batch_count % 1 == 0) {
                    printf("batch_count = %d : %lf\n", batch_count, Loss_batch);
                }
                fprintf(fp, "%d,%lf\n", batch_count, Loss_batch);
            } while (fabs(Loss_batch) > LOSS_MIN && batch_count < N);

            fclose(fp);

            //学習後のパラメータを出力するファイルを開く
            fp = fopen("w_batch.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }

            for (i = 0; i <= ll_param.num_unit[0]; i++) {
                for (j = 0; j <= ll_param.num_unit[1]; j++) {
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }
            }

            fclose(fp);


            if (fabs(Loss_batch) < LOSS_MIN) {
                printf("損失の大きさが一定値を下回りました。\n");
                printf("損失の大きさ : %lf\n", Loss_batch);
                printf("繰り返し回数は%d\n", batch_count);
            }
            else if (batch_count >= N) {
                printf("繰り返し回数が一定数を超えました。\n");
                printf("損失の大きさ : %lf\n", Loss_batch);
                printf("繰り返し回数は%d\n", batch_count);
            }
            else {
                printf("損失の大きさ : %lf\n", Loss_batch);
                printf("繰り返し回数は%d\n", batch_count);
            }
            printf("**************************************************\n");
            break;


        /***************************************************************************************************************************************/
        case 'd':
            /**************************TAの一括学習*************************************/
            printf("TAの定数beta（０～１）を入力してください：\n");
            scanf("%lf", &beta);

            printf("TAの学習回数tfを入力してください：\n");
            scanf("%d", &tf);

            printf("TAのサンプリング時間を入力してください：\n");
            scanf("%lf", &sampling_time);
            printf("**************************************************\n");

            //教師データの非線形変換
            Non_linear_tranform(ll_param, train_data, output_x, train_num);

            //教師データをシャッフルする
            for (i = 0; i < train_num; i++) {
                n_random = rand() % train_num;
                tmp = output_x[i];
                output_x[i] = output_x[n_random];
                output_x[n_random] = tmp;
            }

            printf("TAの逐次学習の処理を始めます．\n");
            //損失関数を出力するファイルを開く
            fp = fopen("loss_seq.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }
            seq_count = 0;  //カウントの初期化
            do {
                Loss_seq = 0.0;

                //教師データについて一つずつ順伝搬・逆伝搬・重みの更新を行う
                for (i = 0; i < train_num; i++) {
                    //順伝搬
                    forward(ll_param, output_x[i], w, layer_in[i], layer_out[i]);
                    J[i] = Cost_Function(layer_out[i][2], t[i], ll_param.output_layer_size);
                    Loss_seq += J[i];

                    if (seq_count == 0 && i == 0) {
                        J0 = J[i];
                    }

                    /*
                    if (i % 10 == 0) {
                        printf("i = %d : %lf\n", i, Loss_seq);
                    }
                    */

                    //重みの更新
                    TA_update_w(ll_param, w, t[i], layer_out[i], J0, beta, tf, J[i], sampling_time);
                }
                //カウント
                seq_count++;

                if (seq_count % 100 == 0) {
                    printf("seq_count = %d : %lf\n", seq_count, Loss_seq);
                }

                fprintf(fp, "%d,%lf\n", seq_count, Loss_seq);
            } while ((fabs(Loss_seq) > LOSS_MIN) && (seq_count < N));

            fclose(fp);

            //学習後のパラメータを出力するファイルを開く
            fp = fopen("w_seq.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }

            for (i = 0; i <= ll_param.num_unit[0]; i++) {
                for (j = 0; j <= ll_param.num_unit[1]; j++) {
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }
            }

            fclose(fp);

            if (fabs(Loss_seq) < LOSS_MIN) {
                printf("損失の大きさが一定値を下回りました。\n");
                printf("損失の大きさ : %lf\n", Loss_seq);
                printf("繰り返し回数は%d\n", seq_count);
            }
            else if (seq_count > N) {
                printf("繰り返し回数が一定数を超えました。\n");
                printf("損失の大きさ : %lf\n", Loss_seq);
                printf("繰り返し回数は%d\n", seq_count);
            }
            else {
                printf("損失の大きさ : %lf\n", Loss_seq);
                printf("繰り返し回数は%d\n", seq_count);
            }
            printf("**************************************************\n");
            break;


        /***************************************************************************************************************************************/
        case 'e':

        if (e_count != 0) {
            printf("**************************************************\n");
            printf("[y] 前回の未学習データを使用する\n");
            printf("[n] 新しいの未学習データを使用する\n");
            printf("**************************************************\n");
            printf("キーを入力して機能を選択してください：\n");
            key = getch();
            printf("**************************************************\n");
        }

        switch (key)
        {
        case 'e':
        case 'n':
            printf("未学習データのファイル名を入力してください：\n");
            scanf("%s", filename);

            printf("未学習データのデータ数を入力してください：\n");
            scanf("%d", &dis_num);

            /*************************************未学習データのパラメータ設定**************************************/

        //dis_layer_out
            if ((dis_layer_out = (double***)malloc((dis_num + 1) * sizeof(double**))) == NULL) {
                exit(-1);
            }

            for (i = 0; i <= dis_num; i++) {
                if ((dis_layer_out[i] = (double**)malloc((LL_N) * sizeof(double*))) == NULL) {
                    exit(-1);
                }

                for (j = 0; j < LL_N; j++) {
                    if ((dis_layer_out[i][j] = (double*)malloc((ll_param.num_unit[j] + 1) * sizeof(double))) == NULL) {
                        exit(-1);
                    }

                    for (k = 0; k <= ll_param.num_unit[j]; k++) {
                        dis_layer_out[i][j][k] = 0.0;
                    }
                }
            }

            //dis_layer_in
            if ((dis_layer_in = (double***)malloc((dis_num + 1) * sizeof(double**))) == NULL) {
                exit(-1);
            }

            for (i = 0; i <= dis_num; i++) {
                if ((dis_layer_in[i] = (double**)malloc((LL_N) * sizeof(double*))) == NULL) {
                    exit(-1);
                }

                for (j = 0; j < LL_N; j++) {
                    if ((dis_layer_in[i][j] = (double*)malloc((ll_param.num_unit[j] + 1) * sizeof(double))) == NULL) {
                        exit(-1);
                    }

                    for (k = 0; k <= ll_param.num_unit[j]; k++) {
                        dis_layer_in[i][j][k] = 0.0;
                    }
                }
            }

            //dis_t
            if ((dis_t = (double**)malloc((dis_num + 1) * sizeof(double*))) == NULL) {
                exit(-1);
            }

            for (i = 0; i <= dis_num; i++) {
                if ((dis_t[i] = (double*)malloc((ll_param.output_layer_size + 1) * sizeof(double))) == NULL) {
                    exit(-1);
                }
            }

            for (i = 0; i <= dis_num; i++) {
                for (j = 0; j <= ll_param.output_layer_size; j++) {
                    dis_t[i][j] = 0.0;
                }
            }

            //unlearn_data
            if ((unlearn_data = (double**)malloc((dis_num + 1) * sizeof(double*))) == NULL) {
                exit(-1);
            }

            for (i = 0; i <= dis_num; i++) {
                if ((unlearn_data[i] = (double*)malloc((ll_param.input_layer_size + 1) * sizeof(double))) == NULL) {
                    exit(-1);
                }
            }

            //dis_output_x
            if ((dis_output_x = (double**)malloc((dis_num + 1) * sizeof(double*))) == NULL) {
                exit(-1);
            }

            for (i = 0; i <= dis_num; i++) {
                if ((dis_output_x[i] = (double*)malloc((ll_param.num_unit[0] + 1) * sizeof(double))) == NULL) {
                    exit(-1);
                }
            }


            //未学習データ
            //ファイルオープン
            if ((fp = fopen(filename, "r")) == NULL) {
                printf("ファイルを開けませんでした．\n");
                return -1;
            }
            //データ読み込み
            //printf("未学習データ\n");
            i = 0;
            while ((fscanf(fp, "%lf,%lf,%lf,%lf", &unlearn_data[i][1], &unlearn_data[i][2], &unlearn_data[i][3], &unlearn_data[i][4])) != EOF) {
                //printf("%.1lf %.1lf\n", unlearn_data[i][1], unlearn_data[i][2]);
                i++;
            }
            //printf("\n");
            fclose(fp);

            printf("未学習データの正解データのファイル名を入力してください：\n");
            scanf("%s", filename);

            //正解データ
            if ((fp = fopen(filename, "r")) == NULL) {
                printf("ファイルを開けませんでした．\n");
                return -1;
            }
            //データ読み込み
            //printf("正解データ\n");
            i = 0;
            while ((fscanf(fp, "%lf,%lf,%lf,%lf", &dis_t[i][1], &dis_t[i][2], &dis_t[i][3], &dis_t[i][4])) != EOF) {
                //printf("%.1lf %.1lf %.1lf %.1lf\n", dis_t[i][1], dis_t[i][2], dis_t[i][3], dis_t[i][4]);
                i++;
            }
            //printf("\n");
            fclose(fp);

            Non_linear_tranform(ll_param, unlearn_data, dis_output_x, dis_num);

            break;

        case 'y':
            break;

        default:
            break;
        }


        for(i = 0; i < dis_num; i++){
            forward(ll_param, dis_output_x[i], w, dis_layer_in[i], dis_layer_out[i]);
        }

        printf("未学習データの出力\n");

        //学習後のパラメータを出力するファイルを開く
        fp = fopen("dis_out.csv", "w");
        if (fp == NULL) {
            printf("ファイルが開けません\n");
            return -1;
        }

        for(i = 0; i < dis_num; i++){
            printf("%d :", i+1);
            for(j = 1; j <= ll_param.output_layer_size; j++){
                printf(" %lf", dis_layer_out[i][2][j]);
            }
            fprintf(fp, "%d,%lf,%lf,%lf,%lf\n", i, dis_layer_out[i][2][1], dis_layer_out[i][2][2], dis_layer_out[i][2][3], dis_layer_out[i][2][4]);

            printf("\n");
        }

        fclose(fp);

        printf("\n");

        printf("正解率は：");
        correct_rate = Accuracy(ll_param, dis_layer_out, dis_t, dis_num);
        printf("%lf\n", correct_rate);

        e_count++;

         printf("**************************************************\n");
         break;


        /***************************************************************************************************************************************/
        case 'f':
            printf("パラメータ（重み，バイアス）をリセットします\n\n");

            srand((unsigned int)time(NULL));

            for(i = 0; i <= ll_param.num_unit[0]; i++) {
                for(j = 0; j <= ll_param.num_unit[1]; j++) {
                    if (j == ll_param.num_unit[1]) {
                        w[i][j] = 0.0;
                    }
                    else {
                        w[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  //乱数でwを初期化
                    }
                }
            }

            //学習前のパラメータを出力するファイルを開く
            fp = fopen("w.csv", "w");
            if (fp == NULL) {
                printf("ファイルが開けません\n");
                return -1;
            }

            for (i = 1; i <= ll_param.num_unit[0]; i++) {
                printf("w[%d] ", i);
                for (j = 1; j <= ll_param.num_unit[1]; j++) {
                    printf(" %9lf", w[i][j]);
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }printf("\n");
            }
            printf("\n");

            fclose(fp);

            printf("パラメータのリセットが完了しました．\n");

            printf("**************************************************\n");
            break;

        case 0x1b:
            printf("プログラムを終了します.\n");
            return -1;
      }
    }

    //メモリ開放
    for(i = 0; i <= ll_param.input_layer_size; i++){
        free(train_data[i]);
    }
    free(train_data);
    for(i = 0; i <= ll_param.input_layer_size; i++){
        free(unlearn_data[i]);
    }
    free(unlearn_data);
    for(i = 0; i <= ll_param.component_num; i++){
        free(w[i]);
    }
    free(w);
    for(i = 0; i <= LL_N; i++){
        for(j = 0; j <= ll_param.output_layer_size; j++){
            free(layer_out[i][j]);
        }

        free(layer_out[i]);
    }
    free(layer_out);
    for(i = 0; i <= LL_N; i++){
        for(j = 0; j <= ll_param.output_layer_size; j++){
            free(layer_in[i][j]);
        }

        free(layer_in[i]);
    }
    free(layer_in);
    for(i = 0; i <= ll_param.output_layer_size; i++){
        free(t[i]);
    }
    free(t);
    for(i = 0; i <= ll_param.input_layer_size; i++){
        free(output_x[i]);
    }
    free(output_x);
    free(ll_param.num_unit);
    free(J);
    for (i = 0; i <= ll_param.output_layer_size; i++) {
        free(dis_t[i]);
    }
    free(dis_t);
    for (i = 0; i <= LL_N; i++) {
        for (j = 0; j <= ll_param.output_layer_size; j++) {
            free(dis_layer_out[i][j]);
        }

        free(dis_layer_out[i]);
    }
    free(dis_layer_out);
    for (i = 0; i <= LL_N; i++) {
        for (j = 0; j <= ll_param.output_layer_size; j++) {
            free(dis_layer_in[i][j]);
        }

        free(dis_layer_in[i]);
    }
    free(dis_layer_in);
    for (i = 0; i <= ll_param.input_layer_size; i++) {
        free(dis_output_x[i]);
    }
    free(dis_output_x);


    return 0;
}
