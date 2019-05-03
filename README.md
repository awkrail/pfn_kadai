# PFN Internship
Preferred Networks インターン選考 2019  
コーディング課題 機械学習・数理分野

# source code
ソースコード, およびレポートは以下のような構成となっています  
```
- README.md
- report.pdf
- prediction.txt
- src
  - datasets
    - train
    - test
  - results
    - 1NN : 1層のニューラルネットワークの結果
      - Adam.pkl
      - MomentumSGD.pkl
      - SGD.pkl
    - 2NN : 2層のニューラルネットワークの結果
      - Adam.pkl
      - MomentumSGD.pkl
      - SGD.pkl
  - code_1.py : 課題1の実行ファイル
  - test_code_1.py : 課題1のテスト実行ファイル
  - code_2.py : 課題2の実行ファイル
  - code_3.py : 課題3の実行ファイル
  - code_4.py : 課題4の実行ファイル
  - common.py : 各課題共通の関数をまとめたファイル
  - utils.py : datasetsからのファイルの読み込みを行う関数をまとめたファイル
  - plot_fig1.py : レポート用の図を出力するファイル : このファイルのみnumpy以外のライブラリを利用しています
```

以下が各ファイルの実行手順です  
課題3は最適化手法をSGDか, momentumSGD(mSGD)かのどちらかを選択できます  
課題4は最適化手法をSGD, momentumSGD(mSGD), adamのいずれか, 層の深さを1, 2のいずれかを選択できます  
```
課題1 : cd src && python code_1.py
課題1(テストコード) : cd src && python test_code_1.py
課題2 : cd src && python code_2.py
課題3 : cd src && python code_3.py -o SGD(or -o mSGD)
課題4 : cd src && python code_4.py -o SGD(or -o mSGD or -o adam) -n 1(or -n 2)
```