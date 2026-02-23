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

    global clip_grad_min
    global clip_grad_max

    global clip_grad_min_ps
    global clip_grad_max_ps

    global batch_size
    global momentum

    ; @data: MLP Architecture

    input_neurons: dq 2
    hidden_neurons: dq 3
    hidden_layers: dq 1
    output_neurons: dq 2

    ; @data: MLP Training

    epochs: dq 50000
    learning_rate: dq 0.01
    enable_dropout: dq 0
    dropout_rate: dq 0.0

    clip_grad_min_ps: dd -1.0, -1.0, -1.0, -1.0
    clip_grad_max_ps: dd 1.0, 1.0, 1.0, 1.0

    clip_grad_min: dd -1.0
    clip_grad_max: dd 1.0

    ; @data: Optimizer Settings

    batch_size: dq 4
    momentum: dq 0.9

; @section: Training Buffers
section .bss
    ; @data: Global buffers

    global weight_grad_base_ptr
    global bias_grad_base_ptr
    global grad_base_ptr

    global weight_velocity
    global bias_velocity

    global activation_buffer
    global delta_buffer

    global output_buffer

    ; @data: Buffers

    weight_grad_base_ptr: resd 16777216
    bias_grad_base_ptr: resd 16777216
    grad_base_ptr: resd 16777216

    weight_velocity: resd 16777216
    bias_velocity: resd 16777216

    activation_buffer: resd 16777216
    delta_buffer: resd 16777216

    output_buffer: resd 256
