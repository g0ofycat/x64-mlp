; @section: MLP Hyperparameters
section .data
    ; @data: Global data

    global input_neurons
    global hidden_neurons
    global hidden_layers
    global output_neurons
    global epochs
    global learning_rate
    global enable_dropout
    global dropout_rate
    global batch_size

    ; @data: MLP Architecture

    input_neurons dq 2
    hidden_neurons dq 3
    hidden_layers dq 1
    output_neurons dq 1

    ; @data: MLP Training

    epochs dq 10000
    learning_rate dd 0.2
    enable_dropout dd 0
    dropout_rate dd 0.0

    ; @data: SGD Settings

    batch_size dq 4

; @section: Training Buffers
section .bss
    ; @data: Global buffers

    global weight_base_ptr
    global bias_base_ptr
    global grad_base_ptr

    global activation_buffer
    global delta_buffer

    global output_buffer

    ; @data: Buffers

    weight_base_ptr resd 16777216
    bias_base_ptr resd 16777216
    grad_base_ptr resd 16777216

    activation_buffer resd 16777216
    delta_buffer resd 16777216

    output_buffer resd 256
