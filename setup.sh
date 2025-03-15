# Install the packages in r1-v .
cd src/

# conda create -n r1-v python=3.11 
# conda activate r1-v
# Addtional modules

pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.3
conda install -c conda-forge mpi4py
