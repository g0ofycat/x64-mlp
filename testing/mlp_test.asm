extern input_neurons
extern hidden_neurons
extern hidden_layers
extern output_neurons
extern epochs
extern learning_rate
extern dropout_rate
extern batch_size

extern mlp_feed_forward
extern mlp_train
extern mlp_back_propagation

extern init_random

extern printf
extern exit

section .text
    global main

main:
    sub rsp, 40

    call init_random

    add rsp, 40

    xor ecx, ecx

    call exit
