source /data/zhongz2/anaconda3/bin/activate th23
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0



python -m serve_controller --host 0.0.0.0 --port 10000
python -m serve_gradio_web_server --controller http://localhost:10000 --model-list-mode reload
python -m serve_model_worker \
    --host 0.0.0.0 \
    --controller-address http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000



python -m serve_controller --host 0.0.0.0 --port 35793
python -m serve_gradio_web_server --controller http://localhost:35793 --model-list-mode reload
python -m serve_model_worker \
    --host 0.0.0.0 \
    --controller http://0.0.0.0:35793 \
    --port 39041 \
    --worker http://0.0.0.0:39041




