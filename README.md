# Neural Programmer Interpreter

This was an attempt to implement [Neural Programmer Interpreter](https://arxiv.org/abs/1511.06279) using [Keras](https://keras.io/).
Unfortunately, I wasn't able to reproduce the paper's results.
Maybe this code will be helpful for someone who wants to implement this type of network in Keras.
I found the [lstm seq2seq example](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py) helpful.

## Usage

1. Generate data
    ```bash
    python -m npi.generate_inputs addition --min 1 --max 20 --examples-per 32 | python -m npi.generate_data --task addition > train.json
    ```

2. Train model
    ```bash
    python -m npi.train --train train.json --encoder addition_encoder.h5 --npi-core npi_core.h5
    ```

3. Inference
    ```bash
    python -m npi.inference --encoder addition_encoder.h5 --npi npi_core.h5 --out inference.json addition --input0 579 --input1 1221
    ```

4. Animate the inference output
    ```bash
    python -m npi.animate inference.json
    ```

    You can also animate a training example since they're in the same format.
    ```bash
    python -m npi.animate train.json --idx 1
    ```

## Notes
1. The authors were able to get their model to learn addition up to 3000 digits with 100% accuracy from examples of 1-20 digits. 
   I wasn't able to get all validation accuracies to 100% when training on examples of 1-20 digits:
    ```bash
    Epoch 18/20
    548/548 [==============================] - 381s 695ms/sample - loss: 0.0095 - stop_layer_loss: 9.4365e-06 - program_key_embedding_layer_loss: 2.5972e-05 - arguments_layer_loss: 0.0084 - stop_layer_weighted_acc: 1.0000 - program_key_embedding_layer_weighted_acc: 1.0000 -
     arguments_layer_weighted_acc: 0.9971 - val_loss: 0.0068 - val_stop_layer_loss: 2.5042e-06 - val_program_key_embedding_layer_loss: 1.6411e-05 - val_arguments_layer_loss: 0.0068 - val_stop_layer_weighted_acc: 1.0000 - val_program_key_embedding_layer_weighted_acc: 1.0000
    - val_arguments_layer_weighted_acc: 0.9974
    Epoch 19/20
    548/548 [==============================] - 381s 695ms/sample - loss: 0.0045 - stop_layer_loss: 3.2207e-06 - program_key_embedding_layer_loss: 1.2200e-05 - arguments_layer_loss: 0.0034 - stop_layer_weighted_acc: 1.0000 - program_key_embedding_layer_weighted_acc: 1.0000 -
     arguments_layer_weighted_acc: 0.9990 - val_loss: 9.0034e-04 - val_stop_layer_loss: 1.4168e-06 - val_program_key_embedding_layer_loss: 7.5872e-06 - val_arguments_layer_loss: 8.8430e-04 - val_stop_layer_weighted_acc: 1.0000 - val_program_key_embedding_layer_weighted_acc:
    l_arguments_layer_weighted_acc: 0.9787
    ```
2. I just used a Dense layer with a softmax activation for the program embedding layer.
3. Using the adaptive sampling from the paper didn't make a difference for me.
4. Training on the addition dataset (32 examples for each input length from 1 to 20) takes a few hours.
