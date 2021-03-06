# Examples using tefla.

1. Imagnet classification using Inception resnet v2
2. Variational autoencoder training
3. Spatial transformer network traning.
4. Unrolled GAN implementation
5. LSTM RNN example
6. Common datasets reading examples. 
7. Autonencoder using Gumbel Softmax
8. Classification networks, Inception resnet, SENET, Xception, RESNET

## Pretrained Weights
1. [VGG19 Trained on ImageNet](https://drive.google.com/file/d/0B9ScQjaDDiwpRnVqZV9JQmh4ZE0/view?usp=sharing)
2. [Inception_Resnet_v2 trained on ImageNet](https://drive.google.com/file/d/0B9ScQjaDDiwpTk1kNDBqT1lKRUU/view?usp=sharing)

# Requirements
```Shell
[GPU version]
pip install tensorflow-gpu>=1.8.0
[CPU version]
pip install tensorflow>=1.8.0

pip install tefla
```

### Train a NLP model on the IMDB dataset
```Shell
[models as CWD]
export PYTHONPATH=$PYTHONPATH:`$PWD`
python nlp/lstm_imdb.py
```
