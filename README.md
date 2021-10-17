# 使用用途
修士研究ではダンボールの中身を入れ替えられていないかを確認する。
そこで、以前取得した特徴量と現在取得した特徴量を比較することで中身が入れ替えられていないかを検証する。
その際に使用していたのがこれらのコードである。

# コードの概要
基本的なコードの流れは以下のようになっている。
1. 60GHzレーダーからデータを取得

2. 以前取得したデータと現在取得したデータを様々な分析手法で比較

# 分析手法の一覧
分析手法は以下のものを使用
- ユークリッド距離
- 相関係数
- KLダイバージェンス
- JSダイバージェンス
- 傾きの遷移
- テンプレートマッチング
- ピーク値の振幅やピーク値の位置



