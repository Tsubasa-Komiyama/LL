# Name

LLGMN (A Log-Linearized Gaussian Mixture Netwaork)
対数線形化された混合正規分布に基づくニューラルネットワーク

ll_func.cにある関数についてはll_func.hをお読みください．

# Usage

まず最初に，コンポーネント数，入力次元数，クラス数を入力してください．
次にa, b, c, d, e, f, escのキーのうち，どれかを押して行う処理を選択してください．
aは一括学習，bは逐次学習を行います.　aとbの最初に学習率を入力してください．
結果の確認はeの学習済みニューロンのテストで行えます.
重みおよびバイアスはfを選択することでリセットできます.
cおよびdはTLですが，現在は値の更新がうまくできません.
終了したい場合は選択画面でescキーを押してください．

# Note

一括学習での損失の推移はloss_batch.csv, 逐次学習での損失の推移はloss_seq.csvに出力されます.
また，パラメータはw.csvに出力され，学習後のパラメータはw_batch.csv, w_seq.csvに出力されます．
未学習データの結果はdis_out.csvに出力されます．
