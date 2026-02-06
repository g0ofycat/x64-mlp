default rel

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

section .data
    input_matrix dd 0.5, 0.8, 0.3, 0.9

    weights dd 0.1, 0.2, 0.3
            dd 0.4, 0.5, 0.6
            dd 0.7, 0.8, 0.9
            dd 0.2, 0.3, 0.4

    biases dd 0.1, 0.2, 0.3

    fmt_output db "Output[%d] = %.4f", 10, 0

section .bss
    output_matrix resd 3

section .text
    global main

main:
    sub rsp, 72

    call init_random

    lea rcx, [input_matrix]
    lea rdx, [weights]
    lea r8, [biases]
    lea r9, [output_matrix]

    mov qword [rsp + 40], 1
    mov qword [rsp + 48], 4
    mov qword [rsp + 56], 3
    mov qword [rsp + 64], 0
    mov qword [rsp + 72], 0

    call mlp_feed_forward

    xor r12, r12

.print_loop:
    cmp r12, 3
    jge .done

    lea rcx, [fmt_output]
    mov rdx, r12

    lea rax, [output_matrix]
    movss xmm0, [rax + r12*4]
    cvtss2sd xmm0, xmm0

    movq r8, xmm0
    movsd xmm0, xmm0

    call printf

    inc r12
    jmp .print_loop

.done:
    add rsp, 72

    xor ecx, ecx
    call exit
