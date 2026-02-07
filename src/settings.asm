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

    input_neurons dq 128
    hidden_neurons dq 512
    hidden_layers dq 2
    output_neurons dq 10

    ; @data: MLP Training

    epochs dq 10000
    learning_rate dd 0.01
    dropout_rate dd 0.1

    ; @data: SGD Settings

    batch_size dq 1

; @section: Training Buffers
section .bss
    ; @data: Global buffers

    global bias_base_ptr
    global grad_base_ptr

    ; @data: Buffers

    bias_base_ptr resd 16777216
    grad_base_ptr resd 16777216
