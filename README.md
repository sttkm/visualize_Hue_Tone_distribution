# visualize Hue and Tone Distribution

画像の色相と色調（明度，彩度）の分布を画像として出力するプロクラム．
色相，明度，彩度は`\{L^{*}a^{*}b^{*}}`色空間においてのものを使用．
`\{L^{*}a^{*}b^{*}}`における色相環の表示や，分布の均しについては工夫してある．
コード説明は今の所，特にしない．

実行は以下．(filename)\_H.jpg，(filename)\_VC.jpgが作成される．
```
python distribution.py (image file)
```

# example

## 1
### origin
![ex1](/example0.jpg)
### Hue distribution
![ex1_H](/exapmle0_H.jpg)
### Tone distribution
![ex1_VC](/example0_VC.jpg)

## 2
### origin
![ex2](/example1.jpg)
### Hue distribution
![ex2_H](/exapmle1_H.jpg)
### Tone distribution
![ex2_VC](/example1_VC.jpg)
