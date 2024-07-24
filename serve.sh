source /data/zhongz2/anaconda3/bin/activate th23
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0



python -m serve_controller --host 0.0.0.0 --port 10000

python -m serve_gradio_web_server --controller http://localhost:10000 --model-list-mode reload










