default rel

extern input_neurons
extern hidden_neurons
extern hidden_layers
extern output_neurons

extern epochs
extern learning_rate
extern enable_dropout
extern dropout_rate
extern batch_size

extern mlp_feed_forward
extern mlp_train
extern mlp_back_propagation

extern weight_base_ptr
extern bias_base_ptr
extern grad_base_ptr

extern init_random
extern init_params

extern printf
extern exit

section .data
    input_tensor dd 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0

    target_tensor dd 0.0, 1.0, 1.0, 0.0

    fmt_output db "Output[%d] = %.4f", 10, 0

section .bss
    output_tensor resd 10000

section .text
    global main

main:
    push rbp
    mov rbp, rsp

    push rsi
    push rbx
    push r12
    push r13
    push r14
    push r15

    sub rsp, 144

    call init_random

    mov rcx, [input_neurons]
    mov rdx, [hidden_neurons]
    mov r8, [hidden_layers]
    mov r9, [output_neurons]
    call init_params

    mov r14, rax
    mov r15, r11

    lea rcx, [input_tensor]
    lea rdx, [target_tensor]
    mov r8, [batch_size]
    mov r9, [input_neurons]
    mov rax, [output_neurons]

    mov [rsp + 40], rax
    mov [rsp + 48], r14
    mov [rsp + 56], r15

    mov rax, [grad_base_ptr]
    mov [rsp + 64], rax
    mov rax, r14
    mov [rsp + 72], rax
    lea rax, [output_tensor]
    mov [rsp + 80], rax
    mov rax, [epochs]
    mov [rsp + 88], rax
    movss xmm0, [learning_rate]
    movss [rsp + 96], xmm0
    mov rax, [enable_dropout]
    mov [rsp + 104], rax
    movss xmm0, [dropout_rate]
    movss [rsp + 112], xmm0
    call mlp_train

    mov r14, rax                ; trained weights
    mov r15, rdx                ; trained biases

    lea rcx, [input_tensor]
    mov rdx, r14                ; trained weights
    mov r8, r15                 ; trained biases
    lea r9, [output_tensor]

    mov rax, [batch_size]
    mov [rsp + 40], rax
    mov rax, [input_neurons]
    mov [rsp + 48], rax
    mov rax, [output_neurons]
    mov [rsp + 56], rax
    mov rax, [enable_dropout]
    mov [rsp + 64], rax
    movss xmm0, [dropout_rate]
    movss [rsp + 72], xmm0
    call mlp_feed_forward

    xor r12, r12
    mov rbx, [output_neurons]

.print_loop:
    cmp r12, rbx
    jge .done

    lea rcx, [fmt_output]
    mov rdx, r12
    lea rax, [output_tensor]
    movss xmm0, [rax + r12*4]
    cvtss2sd xmm0, xmm0
    sub rsp, 32
    call printf
    add rsp, 32

    inc r12
    jmp .print_loop

.done:
    add rsp, 144

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rsi
    pop rbp

    xor ecx, ecx
    call exit
