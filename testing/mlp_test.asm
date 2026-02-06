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
    sub rsp, 40

    call init_random

    lea rcx, [input_matrix]
    lea rdx, [weights]
    lea r8, [biases]
    lea r9, [output_matrix]

    mov qword [rsp + 0], 3
    mov qword [rsp + 8], 4
    mov qword [rsp + 16], 3
    mov qword [rsp + 24], 0
    mov qword [rsp + 32], 0

    call mlp_feed_forward

    xor r12, r12

.print_loop:
    cmp r12, 3
    jge .done

    lea rcx, [fmt_output]   ; format string
    mov rdx, r12            ; 2nd arg (index)

    lea rax, [output_matrix]
    movss xmm0, [rax + r12*4]
    cvtss2sd xmm0, xmm0     ; promote float → double

    movq r8, xmm0           ; 3rd arg integer register
    movsd xmm0, xmm0        ; 3rd arg float register

    call printf

    inc r12
    jmp .print_loop

.done:
    add rsp, 40

    xor ecx, ecx
    call exit
