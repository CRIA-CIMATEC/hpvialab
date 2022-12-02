#/bin/bash

conda create --name demo -y
source activate demo
conda env list
conda install python=3.6.10 -y
pip install zipfile38==0.0.3
conda install numpy=1.17.0 -y
conda install pandas=1.1.3 -y
conda install natsort=7.0.1 -y
pip install heartpy==1.2.7
conda install scipy=1.4.1 -y
conda install matplotlib=3.1.2 -y
pip install opencv-python==3.4.15.55
conda install tk=8.6.8 -y
pip install python-vlc==3.0.12118
pip install mediapipe==0.8.0
pip install vidstab==1.7.3
pip install imutils==0.5.4
pip install dominate==2.3.1
pip install future==0.18.2
pip install scikit-image==0.17.2
pip install scikit-learn==0.24.0
pip install typing-extensions==3.10.0.0
pip install visdom==0.1.8.3
pip install setproctitle==1.1.10
pip install configobj==5.0.6
pip install tqdm==4.23.4
pip install pillow==7.0.0
conda install pytorch=1.4.0 torchvision=0.5.0 numpy=1.17 -c pytorch -y
pip install torchsummary==1.5.1
pip install tensorflow-gpu==1.3.0
pip install progress==1.5

echo "Done!"
