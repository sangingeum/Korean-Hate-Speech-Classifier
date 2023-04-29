# Korean hate speech classifier based on a pretrained model

## Description
The goal of this project is to develop a korean hate speech classification model by fine-tuning a pretrained language model.   

* Pretrained model

__beomi/KcELECTRA-base-v2022__, [hugging face link](https://huggingface.co/beomi/KcELECTRA-base-v2022)


* Dataset

__smilegate-ai/kor_unsmile__, [hugging face link](https://huggingface.co/datasets/smilegate-ai/kor_unsmile)

## Installation

To install the necessary packages, run the following command in your terminal:

    pip3 install -r requirements.txt
    
We recommend installing the CUDA-enabled version of PyTorch, which can be found [here](https://pytorch.org/get-started/locally/)

## Fine-tuning & Testing

To fine-tune the pretrained model and create your own model, execute the following command:

    python train.py model.pt

After N epochs of training, this will create ```model.pt``` file that contains the trained weights.   
To resume training your model from a specific checkpoint, run the following command:

    python train.py model.pt checkpoint.pt
The resulting model will be saved in ```model.pt```.

You can test the accuracy of your model by running the following command:

    python test.py model.pt



## Result

The results of the project will be reported once the classification model is developed and tested.

## license

__beomi/KcELECTRA-base-v2022__, [MIT](https://www.mit.edu/~amini/LICENSE.md)

__smilegate-ai/kor_unsmile__, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

