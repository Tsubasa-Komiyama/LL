#ifndef __INC_LL_FUNC_H
#define __INC_LL_FUNC_H

#include <stdio.h>
#include "parameters.h"


/*!----------------------------------------------------------------------------
 @brif 評価関数

  正解データtとLLGMNの出力yから評価関数t*log(y)を求める
 @param [in] y(double*) 損失を評価するデータ
 @param [in] t(double*) 正解データ
 @param [in] size(int)　yおよびtのサイズ
 @return double 評価関数
 @attention
 @par 更新履歴
   - 2020/4/21
     -基本的な機能の実装 (by Tsubasa Komiyama)

*/

double Cost_Function(double *y, double *t, int size);

/*!----------------------------------------------------------------------------
 @brif 順伝搬の処理を行う関数

  順伝搬における入力層，中間層，出力層での処理を定義
 @param [in] ll_param(LL_PARAM) LL_PARAM構造体のデータ
 @param [in] data(double*) 入力するデータ
 @param [in] w(double***) 重み・バイアス
 @param [in] h(int) 非線形変換後の入力ベクトルの次元
 @param [in] layer_in(double**) 各層の入力
 @param [in] layer_out(double**) 各層の出力
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/22
     -基本的な機能の実装 (by Tsubasa Komiyama)

*/

void forward(LL_PARAM ll_param, double *data, double **w, double **layer_in, double **layer_out);

/*!----------------------------------------------------------------------------
 @brif 重みの更新を行う関数

  重みを更新する. 逐次学習で使用.
 @param [in] ll_param(LL_PARAM) LL_PARAM構造体のデータ
 @param [in] epsilon(double) 学習率
 @param [in] w(double***) 重み
 @param [in] h(int*) 各層の素子数を格納してある配列
 @param [in] dE_dw(double***) 逆伝搬で得た損失の微分
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/23
     -基本的な機能の実装 (by Tsubasa Komiyama)

*/

void update_w(LL_PARAM ll_param, double epsilon, double **w, double *t, double **layer_out);

/*!----------------------------------------------------------------------------
 @brif 重みの更新を行う関数

  全部の教師データの損失から重みを更新する. 一括学習で使用.
 @param [in] ll_param(LL_PARAM) LL_PARAM構造体のデータ
 @param [in] epsilon(double) 学習率
 @param [in] w(double***) 重み
 @param [in] h(int*) 各層の素子数を格納してある配列
 @param [in] dE_dw_t(double***) dE_dwの合計
 @param [in] batch_size(int) データ数
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/23
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

void batch_update_w(LL_PARAM ll_param, double epsilon, double** w, double** t, double*** layer_out, int batch_size, double*** layer_in);

/*!----------------------------------------------------------------------------
 @brif 構造体LL_PARAMの初期化を行う関数

  LL_PARAMのメモリ確保，初期化を行う
 @param [in] ll_param(LL_PARAM) LL_PARAM構造体のデータ
 @return LL_PARAM
 @attention
 @par 更新履歴
   - 2020/4/21
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

LL_PARAM set_param(LL_PARAM ll_param);

/*!----------------------------------------------------------------------------
 @brif 入力ベクトルの非線形変換

  入力ベクトルxを非線形変換によってLLGMNに適した入力ベクトルXに変換する
 @param [in] input_x(double*) 入力ベクトルx
 @param [out] output_x(double*) 変換後の入力ベクトルX
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/21
     -基本的な機能の実装 (by Tsubasa Komiyama)

*/

void Non_linear_tranform(LL_PARAM ll_param, double** input_x, double** output_x);

/*!----------------------------------------------------------------------------
 @brif ターミナルアトラクタ

  ターミナルラーニング用の更新関数
 @param [in] input_x(double*) 入力ベクトルx
 @param [out] output_x(double*) 変換後の入力ベクトルX
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/25
     -基本的な機能の実装 (by Tsubasa Komiyama)

*/

void Non_linear_tranform(LL_PARAM ll_param, double** input_x, double** output_x);




#endif
