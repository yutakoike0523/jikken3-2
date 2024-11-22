from ultralytics import YOLO

# 事前学習済みモデルをロード
model = YOLO('yolo11n.pt')  # yolo11n.pt の場所を指定

# トレーニングデータセットを指定
if __name__ == '__main__':
    model.train(
        data='C:\\Users\\kyuta\\programing\\jikken\\jikken3-2\\data.yaml',  # データセットの設定ファイル (下記参照)
        epochs=50,                         # エポック数
        imgsz=640,                         # 入力画像サイズ
        batch=16,                          # バッチサイズ
        device=0                           # GPU (0 = GPUを使用, 'cpu' = CPU)
    )
