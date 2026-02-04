; @section: MLP Hyperparameters
section .data
    ; @data: Global data

    global input_neurons
    global hidden_neurons
    global hidden_layers
    global output_neurons
    global epochs
    global learning_rate
    global dropout_rate
    global batch_size

    ; @data: MLP Architecture

    input_neurons dw 128
    hidden_neurons dw 512
    hidden_layers dw 2
    output_neurons dw 10

    ; @data: MLP Training

    epochs dw 10000
    learning_rate dd 0.01
    dropout_rate dd 0.1

    ; @data: SGD Settings

    batch_size db 32
