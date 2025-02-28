# YuE-server

## 環境構築
３種類のファイルを
/YuE-exllamav2/src/yue/
フィルダーにコピー
オリジナルの環境を設定後、環境を有効化
fastapi
uvicorn
gradio
matplotlib
exllamav2
transformers
をインストール

## サーバ起動
YuE-exllamav2フォルダに移動

python src/yue/yue_q_server_1f2s.py
でサーバを起動

## Gradioアプリ
YuE-serverと同様の環境を有効化し、YuE-exllamav2フォルダに移動

python yue_gui_x.py
で起動
