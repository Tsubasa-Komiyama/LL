#ifndef __INC_LL_FUNC_H
#define __INC_LL_FUNC_H

#include <stdio.h>
#include "parameters.h"


/*!----------------------------------------------------------------------------
 @brif �]���֐�

  �����f�[�^t��LLGMN�̏o��y����]���֐�t*log(y)�����߂�
 @param [in] y(double*) ������]������f�[�^
 @param [in] t(double*) �����f�[�^
 @param [in] size(int)�@y�����t�̃T�C�Y
 @return double �]���֐�
 @attention
 @par �X�V����
   - 2020/4/21
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/

double Cost_Function(double *y, double *t, int size);

/*!----------------------------------------------------------------------------
 @brif ���`���̏������s���֐�

  ���`���ɂ�������͑w�C���ԑw�C�o�͑w�ł̏������`
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @param [in] data(double*) ���͂���f�[�^
 @param [in] w(double**) �d�݁E�o�C�A�X
 @param [in,out] layer_in(double**) �e�w�̓���
 @param [in,out] layer_out(double**) �e�w�̏o��
 @return �Ȃ�
 @attention
 @par �X�V����
   - 2020/4/22
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/

void forward(LL_PARAM ll_param, double *data, double **w, double **layer_in, double **layer_out);

/*!----------------------------------------------------------------------------
 @brif �d�݂̍X�V���s���֐�

  �d�݂��X�V����. �����w�K�Ŏg�p.
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @param [in] epsilon(double) �w�K��
 @param [in,out] w(double**) �d��
 @param [in] t(double*) �����f�[�^
 @param [in] layer_out(double**) �e�w�̏o�͂��i�[���Ă���z��
 @return �Ȃ�
 @attention
 @par �X�V����
   - 2020/4/23
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/

void update_w(LL_PARAM ll_param, double epsilon, double **w, double *t, double **layer_out);

/*!----------------------------------------------------------------------------
 @brif �d�݂̍X�V���s���֐�

  �S���̋��t�f�[�^�̑�������d�݂��X�V����. �ꊇ�w�K�Ŏg�p.
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @param [in] epsilon(double) �w�K��
 @param [in,out] w(double**) �d��
 @param [in] t(double**) �����f�[�^
 @param [in] layer_out(double***) �e�w�̏o�͂��i�[���Ă���z��
 @param [in] batch_size(int) �f�[�^��
 @return �Ȃ�
 @attention
 @par �X�V����
   - 2020/4/23
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)
*/

void batch_update_w(LL_PARAM ll_param, double epsilon, double** w, double** t, double*** layer_out, int batch_size);

/*!----------------------------------------------------------------------------
 @brif �\����LL_PARAM�̏��������s���֐�

  LL_PARAM�̃������m�ہC���������s��
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @return LL_PARAM
 @attention
 @par �X�V����
   - 2020/4/21
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)
*/

LL_PARAM set_param(LL_PARAM ll_param);

/*!----------------------------------------------------------------------------
 @brif ���̓x�N�g���̔���`�ϊ�

  ���̓x�N�g��x�����`�ϊ��ɂ����LLGMN�ɓK�������̓x�N�g��X�ɕϊ�����
 @param [in] input_x(double*) ���̓x�N�g��x
 @param [out] output_x(double*) �ϊ���̓��̓x�N�g��X
 @return �Ȃ�
 @attention
 @par �X�V����
   - 2020/4/21
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/

void Non_linear_tranform(LL_PARAM ll_param, double** input_x, double** output_x);

/*!----------------------------------------------------------------------------
 @brif �����w�K�p�^�[�~�i���A�g���N�^

  �����w�K�ɂ�����^�[�~�i�����[�j���O�p�̍X�V�֐�
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @param [in] epsilon(double) �w�K��
 @param [in,out] w(double**) �d��
 @param [in] t(double*) �����f�[�^
 @param [in] layer_out(double**) �e�w�̏o�͂��i�[���Ă���z��
 @param [in] J0(double) �]���֐��v�̏����l
 @param [in] beta(double) �O�`�P�̒萔
 @param [in] tf(int) ��������
 @param [in] delta_t(double) �T���v�����O����
 @param [in] J(double) �e�w�̏o�͂��i�[���Ă���z��
 @return �Ȃ�
 @attention
 @par �X�V����
   - 2020/4/26
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/

void TA_update_w(LL_PARAM ll_param, double** w, double* t, double** layer_out, double J0, double beta, int tf, double J, double delta_t);

/*!----------------------------------------------------------------------------
 @brif �ꊇ�w�K�p�^�[�~�i���A�g���N�^

  �ꊇ�w�K�ɂ�����^�[�~�i�����[�j���O�p�̍X�V�֐�
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @param [in] epsilon(double) �w�K��
 @param [in,out] w(double**) �d��
 @param [in] t(double**) �����f�[�^
 @param [in] layer_out(double***) �e�w�̏o�͂��i�[���Ă���z��
 @param [in] J0(double) �]���֐��v�̏����l
 @param [in] beta(double) �O�`�P�̒萔
 @param [in] tf(int) ��������
 @param [in] delta_t(double) �T���v�����O����
 @param [in] J(double*) �e�w�̏o�͂��i�[���Ă���z��
 @param [in] batch_size(int) �f�[�^��
 @return �Ȃ�
 @attention
 @par �X�V����
   - 2020/4/26
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/

void TA_batch_update_w(LL_PARAM ll_param, double** w, double** t, double*** layer_out, double J0, double beta, int tf, double delta_t, double* J, int batch_size);


/*!----------------------------------------------------------------------------
 @brif �ꊇ�w�K�p�^�[�~�i���A�g���N�^

  �ꊇ�w�K�ɂ�����^�[�~�i�����[�j���O�p�̍X�V�֐�
 @param [in] ll_param(LL_PARAM) LL_PARAM�\���̂̃f�[�^
 @param [in] layer_out(double***)�@�e�w�̏o��
 @param [in] t(double**) �����f�[�^
 @param [in] data_num(int) �f�[�^��
 @return double ����
 @attention
 @par �X�V����
   - 2020/4/27
     -��{�I�ȋ@�\�̎��� (by Tsubasa Komiyama)

*/
double Accuracy(LL_PARAM ll_param, double*** layer_out, double** t, int data_num);

#endif
