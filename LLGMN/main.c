#include <stdio.h>
#include "ll_func.h"
#include "parameters.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <conio.h>
#pragma warning(disable : 4996)

#define FILENAME 30     //�t�@�C���̍ő啶����

int main(void){
    int i, j, k;        //����ϐ�
    int key;  //�Θb�p�I����
    LL_PARAM ll_param;  //�\����
    FILE *fp;
    char filename[FILENAME];    //�t�@�C����
    double Loss_batch;
    double Loss_seq;        //�����֐�
    double epsilon;     //�w�K��
    int batch_count;    //�ꊇ�w�K��
    int seq_count;      //�����w�K��
    double beta;        //TA�̒萔
    int tf;             //TA�̊w�K��
    double sampling_time; //�T���v�����O����
    double J0;          //�]���֐��̏����l
    int n_random;                   //���̓x�N�g���̃V���b�t���Ɏg�p
    double* tmp = NULL;             //���̓x�N�g���̃V���b�t���Ɏg�p
    double* tmp_t = NULL;           //�����f�[�^�̃V���b�t���Ɏg�p
    double correct_rate;            //����

    int train_num;                  //���t�f�[�^�̃f�[�^��
    int dis_num;                    //���w�K�f�[�^�̃f�[�^��

    int e_count = 0;                //key 'e' ��I��������


    /**************************�w���E�f�q���̐ݒ�*****************************/

    printf("�R���|�[�l���g������͂��Ă��������F\n");
    scanf("%d", &ll_param.component_num);

    printf("���͎���������͂��Ă��������F\n");
    scanf("%d", &ll_param.input_layer_size);

    printf("�N���X������͂��Ă��������F\n");
    scanf("%d", &ll_param.output_layer_size);

    printf("���t�f�[�^�̃t�@�C��������͂��Ă��������F\n");
    scanf("%s", filename);

    printf("���t�f�[�^�̃f�[�^������͂��Ă��������F\n");
    scanf("%d", &train_num);



    //ll_param.component_num = 2;
    //ll_param.input_layer_size = 2;
    //ll_param.output_layer_size = 4;

    /**************************�e��p�����[�^�̐ݒ�*****************************/
    ll_param = set_param(ll_param);

    double **train_data = NULL;     //���̓f�[�^
    double **w = NULL;              //�d��
    double ***layer_in = NULL;       //�e�w�̓���
    double ***layer_out = NULL;      //�e�w�̏o��
    double **t = NULL;              //�����f�[�^
    double **unlearn_data;          //���w�K�f�[�^
    double **output_x = NULL;       //����`�ϊ���̓��̓x�N�g��
    double *J = NULL;
    double** dis_t = NULL;               //���w�K�f�[�^�̐����f�[�^
    double*** dis_layer_in = NULL;       //���w�K�f�[�^�̊e�w�̓���
    double*** dis_layer_out = NULL;      //���w�K�f�[�^�̊e�w�̏o��
    double** dis_output_x = NULL;        //���w�K�f�[�^�̔���`�ϊ���̓��̓x�N�g��


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

    //�����̏�����
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
                w[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  //������w��������
            }
        }
    }

    //�w�K�O�̃p�����[�^���o�͂���t�@�C�����J��
    fp = fopen("w.csv", "w");
    if( fp == NULL ){
        printf( "�t�@�C�����J���܂���\n");
        return -1;
    }

    printf("�d��\n");

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


    /**************************���t�f�[�^�E�����f�[�^�̓ǂݍ���*****************************/

    //���t�f�[�^
    //�t�@�C���I�[�v��
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("�t�@�C�����J���܂���ł����D\n");
        return -1;
    }
    //�f�[�^�ǂݍ���
    //printf("���t�f�[�^\n");
    i = 0;
    while ((fscanf(fp, "%lf,%lf,%lf,%lf", &train_data[i][1], &train_data[i][2], &train_data[i][3], &train_data[i][4])) != EOF) {
      //printf("%.1lf %.1lf\n", train_data[i][1], train_data[i][2]);
      i++;
    }
    //printf("\n");
    fclose(fp);

    printf("�����f�[�^�̃t�@�C��������͂��Ă��������F\n");
    scanf("%s", filename);

    //�����f�[�^
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("�t�@�C�����J���܂���ł����D\n");
        return -1;
    }
    //�f�[�^�ǂݍ���
    //printf("�����f�[�^\n");
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

    /**************************�V�X�e��*****************************/


    printf("**************************************************\n");
    printf("LLGMN\n");
    printf("**************************************************\n");


    while(1){

      printf("[a] �ꊇ�X�V�w�K�@\n");
      printf("[b] �����X�V�w�K�@\n");
      printf("[c] TA��p�����ꊇ�X�V�w�K�@\n");
      printf("[d] TA��p���������X�V�w�K�@\n");
      printf("[e] �w�K�ς݃j���[�����̃e�X�g\n");
      printf("[f] �p�����[�^�̃��Z�b�g\n");
      printf("[ESC] �v���O�����I��\n");

      printf("**************************************************\n");
      printf("�L�[����͂��ċ@�\��I�����Ă��������F\n");
      key = getch();
      printf("**************************************************\n");

      switch (key) {

        /***************************************************************************************************************************************/
        case 'a':
        /**************************�ꊇ�w�K*************************************/
        printf("�w�K������͂��Ă��������F\n");
        scanf("%lf", &epsilon);
        printf("**************************************************\n");

        //���t�f�[�^�̔���`�ϊ�
        Non_linear_tranform(ll_param, train_data, output_x, train_num);

        printf("�ꊇ�w�K�̏������n�߂܂��D\n");
        //�������i�[����t�@�C�����J��
        fp = fopen("loss_batch.csv", "w");
        if( fp == NULL ){
            printf( "�t�@�C�����J���܂���\n");
            return -1;
        }
        batch_count = 0;    //�J�E���g�̏�����
        do{
            //���t�f�[�^�ɂ��Ĉ�����`���E�t�`�����s���C�d�݂̍X�V�̓G�|�b�N���Ƃɍs��
            Loss_batch = 0.0;
            for(i = 0; i < train_num; i++){
                //���`��
                forward(ll_param, output_x[i], w, layer_in[i], layer_out[i]);
                Loss_batch += Cost_Function(layer_out[i][2], t[i], ll_param.output_layer_size);

                /*
                if (i % 10 == 0) {
                    printf("i = %d : %lf\n", i, Loss_batch);
                }
                */
            }

            //�d�݂̍X�V
            batch_update_w(ll_param, epsilon, w, t, layer_out, train_num);

            /*
            //�X�V��̃p�����[�^���o�͂���t�@�C�����J��
            fp = fopen("w_test.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }

            for (i = 0; i <= ll_param.num_unit[0]; i++) {
                for (j = 0; j <= ll_param.num_unit[1]; j++) {
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }
            }

            fclose(fp);

            //layer_in���o�͂���t�@�C�����J��
            fp = fopen("in_test.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }

            for (i = 0; i < train_num; i++) {
                fprintf(fp, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, layer_in[i][0][1], layer_in[i][0][2], layer_in[i][0][3], layer_in[i][0][4], layer_in[i][0][5], layer_in[i][0][6], layer_in[i][1][1], layer_in[i][1][2], layer_in[i][1][3], layer_in[i][1][4], layer_in[i][1][5], layer_in[i][1][6], layer_in[i][1][7], layer_in[i][1][8], layer_in[i][2][1], layer_in[i][2][2], layer_in[i][2][3], layer_in[i][2][4]);
            }

            fclose(fp);

            //layer_out���o�͂���t�@�C�����J��
            fp = fopen("out_test.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }

            for (i = 0; i < train_num; i++) {
                fprintf(fp, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", i, layer_out[i][0][1], layer_out[i][0][2], layer_out[i][0][3], layer_out[i][0][4], layer_out[i][0][5], layer_out[i][0][6], layer_out[i][1][1], layer_out[i][1][2], layer_out[i][1][3], layer_out[i][1][4], layer_out[i][1][5], layer_out[i][1][6], layer_out[i][1][7], layer_out[i][1][8], layer_out[i][2][1], layer_out[i][2][2], layer_out[i][2][3], layer_out[i][2][4]);
            }

            fclose(fp);
            */


            //�J�E���g
            batch_count++;

            if(batch_count % 1 == 0){
                printf("batch_count = %d : %lf\n", batch_count, Loss_batch);
            }
            fprintf(fp, "%d,%lf\n", batch_count, Loss_batch);
        }while(fabs(Loss_batch) > LOSS_MIN && batch_count < N);

        fclose(fp);

        //�w�K��̃p�����[�^���o�͂���t�@�C�����J��
        fp = fopen("w_batch.csv", "w");
        if( fp == NULL ){
            printf( "�t�@�C�����J���܂���\n");
            return -1;
        }

        for(i = 0; i <= ll_param.num_unit[0]; i++) {
            for(j = 0; j <= ll_param.num_unit[1]; j++) {
                fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
            }
        }

        fclose(fp);


          if(fabs(Loss_batch) < LOSS_MIN){
            printf("�����̑傫�������l�������܂����B\n");
            printf("�����̑傫�� : %lf\n", Loss_batch);
            printf("�J��Ԃ��񐔂�%d\n", batch_count);
        }else if(batch_count >= N){
            printf("�J��Ԃ��񐔂���萔�𒴂��܂����B\n");
            printf("�����̑傫�� : %lf\n", Loss_batch);
            printf("�J��Ԃ��񐔂�%d\n", batch_count);
          }else{
            printf("�����̑傫�� : %lf\n", Loss_batch);
            printf("�J��Ԃ��񐔂�%d\n", batch_count);
          }
          printf("**************************************************\n");
          break;


        /***************************************************************************************************************************************/
        case 'b':
        /**************************�����w�K*************************************/
        printf("�w�K������͂��Ă��������F\n");
        scanf("%lf", &epsilon);
        printf("**************************************************\n");

        //���t�f�[�^�̔���`�ϊ�
        Non_linear_tranform(ll_param, train_data, output_x, train_num);

        srand((unsigned int)time(NULL));

        //���t�f�[�^���V���b�t������
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

        printf("�����w�K�̏������n�߂܂��D\n");
        //�����֐����o�͂���t�@�C�����J��
        fp = fopen("loss_seq.csv", "w");
        if( fp == NULL ){
            printf( "�t�@�C�����J���܂���\n");
            return -1;
        }
        seq_count = 0;  //�J�E���g�̏�����
        do{
            Loss_seq = 0.0;

            //���t�f�[�^�ɂ��Ĉ�����`���E�t�`���E�d�݂̍X�V���s��
            for(i = 0; i < train_num; i++){
                //���`��
                forward(ll_param, output_x[i], w, layer_in[i], layer_out[i]);
                Loss_seq += Cost_Function(layer_out[i][2], t[i], ll_param.output_layer_size);

                /*
                if (i % 10 == 0) {
                    printf("i = %d : %lf\n", i, Loss_seq);
                }
                */

                //�d�݂̍X�V
                update_w(ll_param, epsilon, w, t[i], layer_out[i]);
            }
            //�J�E���g
            seq_count++;

            if(seq_count % 1 == 0){
                printf("seq_count = %d : %lf\n", seq_count, Loss_seq);
            }

            fprintf(fp, "%d,%lf\n", seq_count, Loss_seq);
        }while((fabs(Loss_seq) > LOSS_MIN) && (seq_count < N));

        fclose(fp);

        //�w�K��̃p�����[�^���o�͂���t�@�C�����J��
        fp = fopen("w_seq.csv", "w");
        if( fp == NULL ){
            printf( "�t�@�C�����J���܂���\n");
            return -1;
        }

        for(i = 0; i <= ll_param.num_unit[0]; i++) {
            for(j = 0; j <= ll_param.num_unit[1]; j++) {
                fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
            }
        }

        fclose(fp);

          if(fabs(Loss_seq) < LOSS_MIN){
            printf("�����̑傫�������l�������܂����B\n");
            printf("�����̑傫�� : %lf\n", Loss_seq);
            printf("�J��Ԃ��񐔂�%d\n", seq_count);
        }else if(seq_count > N){
            printf("�J��Ԃ��񐔂���萔�𒴂��܂����B\n");
            printf("�����̑傫�� : %lf\n", Loss_seq);
            printf("�J��Ԃ��񐔂�%d\n", seq_count);
          }else{
            printf("�����̑傫�� : %lf\n", Loss_seq);
            printf("�J��Ԃ��񐔂�%d\n", seq_count);
          }
          printf("**************************************************\n");
          break;



        /***************************************************************************************************************************************/
        case 'c':
            /**************************TA�̈ꊇ�w�K*************************************/
            printf("TA�̒萔beta�i�O�`�P�j����͂��Ă��������F\n");
            scanf("%lf", &beta);

            printf("TA�̊w�K��tf����͂��Ă��������F\n");
            scanf("%d", &tf);

            printf("TA�̃T���v�����O���Ԃ���͂��Ă��������F\n");
            scanf("%lf", &sampling_time);
            printf("**************************************************\n");

            //���t�f�[�^�̔���`�ϊ�
            Non_linear_tranform(ll_param, train_data, output_x, train_num);

            printf("TA��p�����ꊇ�w�K�̏������n�߂܂��D\n");
            //�������i�[����t�@�C�����J��
            fp = fopen("loss_batch.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }
            batch_count = 0;    //�J�E���g�̏�����
            do {
                //���t�f�[�^�ɂ��Ĉ�����`���E�t�`�����s���C�d�݂̍X�V�̓G�|�b�N���Ƃɍs��
                Loss_batch = 0.0;
                for (i = 0; i < train_num; i++) {
                    //���`��
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

                //�d�݂̍X�V
                TA_batch_update_w(ll_param, w, t, layer_out, J0, beta, tf, sampling_time, J, train_num);

                //�J�E���g
                batch_count++;

                if (batch_count % 1 == 0) {
                    printf("batch_count = %d : %lf\n", batch_count, Loss_batch);
                }
                fprintf(fp, "%d,%lf\n", batch_count, Loss_batch);
            } while (fabs(Loss_batch) > LOSS_MIN && batch_count < N);

            fclose(fp);

            //�w�K��̃p�����[�^���o�͂���t�@�C�����J��
            fp = fopen("w_batch.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }

            for (i = 0; i <= ll_param.num_unit[0]; i++) {
                for (j = 0; j <= ll_param.num_unit[1]; j++) {
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }
            }

            fclose(fp);


            if (fabs(Loss_batch) < LOSS_MIN) {
                printf("�����̑傫�������l�������܂����B\n");
                printf("�����̑傫�� : %lf\n", Loss_batch);
                printf("�J��Ԃ��񐔂�%d\n", batch_count);
            }
            else if (batch_count >= N) {
                printf("�J��Ԃ��񐔂���萔�𒴂��܂����B\n");
                printf("�����̑傫�� : %lf\n", Loss_batch);
                printf("�J��Ԃ��񐔂�%d\n", batch_count);
            }
            else {
                printf("�����̑傫�� : %lf\n", Loss_batch);
                printf("�J��Ԃ��񐔂�%d\n", batch_count);
            }
            printf("**************************************************\n");
            break;


        /***************************************************************************************************************************************/
        case 'd':
            /**************************TA�̈ꊇ�w�K*************************************/
            printf("TA�̒萔beta�i�O�`�P�j����͂��Ă��������F\n");
            scanf("%lf", &beta);

            printf("TA�̊w�K��tf����͂��Ă��������F\n");
            scanf("%d", &tf);

            printf("TA�̃T���v�����O���Ԃ���͂��Ă��������F\n");
            scanf("%lf", &sampling_time);
            printf("**************************************************\n");

            //���t�f�[�^�̔���`�ϊ�
            Non_linear_tranform(ll_param, train_data, output_x, train_num);

            //���t�f�[�^���V���b�t������
            for (i = 0; i < train_num; i++) {
                n_random = rand() % train_num;
                tmp = output_x[i];
                output_x[i] = output_x[n_random];
                output_x[n_random] = tmp;
            }

            printf("TA�̒����w�K�̏������n�߂܂��D\n");
            //�����֐����o�͂���t�@�C�����J��
            fp = fopen("loss_seq.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }
            seq_count = 0;  //�J�E���g�̏�����
            do {
                Loss_seq = 0.0;

                //���t�f�[�^�ɂ��Ĉ�����`���E�t�`���E�d�݂̍X�V���s��
                for (i = 0; i < train_num; i++) {
                    //���`��
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

                    //�d�݂̍X�V
                    TA_update_w(ll_param, w, t[i], layer_out[i], J0, beta, tf, J[i], sampling_time);
                }
                //�J�E���g
                seq_count++;

                if (seq_count % 100 == 0) {
                    printf("seq_count = %d : %lf\n", seq_count, Loss_seq);
                }

                fprintf(fp, "%d,%lf\n", seq_count, Loss_seq);
            } while ((fabs(Loss_seq) > LOSS_MIN) && (seq_count < N));

            fclose(fp);

            //�w�K��̃p�����[�^���o�͂���t�@�C�����J��
            fp = fopen("w_seq.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
                return -1;
            }

            for (i = 0; i <= ll_param.num_unit[0]; i++) {
                for (j = 0; j <= ll_param.num_unit[1]; j++) {
                    fprintf(fp, "%d,%d,%lf\n", i, j, w[i][j]);
                }
            }

            fclose(fp);

            if (fabs(Loss_seq) < LOSS_MIN) {
                printf("�����̑傫�������l�������܂����B\n");
                printf("�����̑傫�� : %lf\n", Loss_seq);
                printf("�J��Ԃ��񐔂�%d\n", seq_count);
            }
            else if (seq_count > N) {
                printf("�J��Ԃ��񐔂���萔�𒴂��܂����B\n");
                printf("�����̑傫�� : %lf\n", Loss_seq);
                printf("�J��Ԃ��񐔂�%d\n", seq_count);
            }
            else {
                printf("�����̑傫�� : %lf\n", Loss_seq);
                printf("�J��Ԃ��񐔂�%d\n", seq_count);
            }
            printf("**************************************************\n");
            break;


        /***************************************************************************************************************************************/
        case 'e':

        if (e_count != 0) {
            printf("**************************************************\n");
            printf("[y] �O��̖��w�K�f�[�^���g�p����\n");
            printf("[n] �V�����̖��w�K�f�[�^���g�p����\n");
            printf("**************************************************\n");
            printf("�L�[����͂��ċ@�\��I�����Ă��������F\n");
            key = getch();
            printf("**************************************************\n");
        }

        switch (key)
        {
        case 'e':
        case 'n':
            printf("���w�K�f�[�^�̃t�@�C��������͂��Ă��������F\n");
            scanf("%s", filename);

            printf("���w�K�f�[�^�̃f�[�^������͂��Ă��������F\n");
            scanf("%d", &dis_num);

            /*************************************���w�K�f�[�^�̃p�����[�^�ݒ�**************************************/

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


            //���w�K�f�[�^
            //�t�@�C���I�[�v��
            if ((fp = fopen(filename, "r")) == NULL) {
                printf("�t�@�C�����J���܂���ł����D\n");
                return -1;
            }
            //�f�[�^�ǂݍ���
            //printf("���w�K�f�[�^\n");
            i = 0;
            while ((fscanf(fp, "%lf,%lf,%lf,%lf", &unlearn_data[i][1], &unlearn_data[i][2], &unlearn_data[i][3], &unlearn_data[i][4])) != EOF) {
                //printf("%.1lf %.1lf\n", unlearn_data[i][1], unlearn_data[i][2]);
                i++;
            }
            //printf("\n");
            fclose(fp);

            printf("���w�K�f�[�^�̐����f�[�^�̃t�@�C��������͂��Ă��������F\n");
            scanf("%s", filename);

            //�����f�[�^
            if ((fp = fopen(filename, "r")) == NULL) {
                printf("�t�@�C�����J���܂���ł����D\n");
                return -1;
            }
            //�f�[�^�ǂݍ���
            //printf("�����f�[�^\n");
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

        printf("���w�K�f�[�^�̏o��\n");

        //�w�K��̃p�����[�^���o�͂���t�@�C�����J��
        fp = fopen("dis_out.csv", "w");
        if (fp == NULL) {
            printf("�t�@�C�����J���܂���\n");
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

        printf("���𗦂́F");
        correct_rate = Accuracy(ll_param, dis_layer_out, dis_t, dis_num);
        printf("%lf\n", correct_rate);

        e_count++;

         printf("**************************************************\n");
         break;


        /***************************************************************************************************************************************/
        case 'f':
            printf("�p�����[�^�i�d�݁C�o�C�A�X�j�����Z�b�g���܂�\n\n");

            srand((unsigned int)time(NULL));

            for(i = 0; i <= ll_param.num_unit[0]; i++) {
                for(j = 0; j <= ll_param.num_unit[1]; j++) {
                    if (j == ll_param.num_unit[1]) {
                        w[i][j] = 0.0;
                    }
                    else {
                        w[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;  //������w��������
                    }
                }
            }

            //�w�K�O�̃p�����[�^���o�͂���t�@�C�����J��
            fp = fopen("w.csv", "w");
            if (fp == NULL) {
                printf("�t�@�C�����J���܂���\n");
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

            printf("�p�����[�^�̃��Z�b�g���������܂����D\n");

            printf("**************************************************\n");
            break;

        case 0x1b:
            printf("�v���O�������I�����܂�.\n");
            return -1;
      }
    }

    //�������J��
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
