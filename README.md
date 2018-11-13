# visualize Hue and Tone Distribution

画像の色相と色調（明度，彩度）の分布を画像として出力するプロクラム．  
色相，明度，彩度はL<sup>\*</sup>a<sup>\*</sup>b<sup>\*</sup>色空間においてのものを使用．  
L<sup>\*</sup>a<sup>\*</sup>b<sup>\*</sup>における色相環の表示や，分布の均しについては工夫してある．  
コード説明は今の所，特にしない．

実行は以下．(filename)\_H.jpg，(filename)\_VC.jpgが作成される．
```
python distribution.py (image file)
```

# example

## 1
### origin
![ex1](./example/ex0.jpg)
### Hue distribution
![ex1_H](./example/ex0_H.jpg)
### Tone distribution
![ex1_VC](./example/ex0_VC.jpg)

## 2
### origin
![ex2](./example/ex1.jpg)
### Hue distribution
![ex2_H](./example/ex1_H.jpg)
### Tone distribution
![ex2_VC](./example/ex1_VC.jpg)
