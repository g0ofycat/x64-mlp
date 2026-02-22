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
extern momentum

extern mlp_feed_forward
extern mlp_train
extern mlp_back_propagation

extern weight_grad_base_ptr
extern bias_grad_base_ptr
extern grad_base_ptr

extern weight_velocity
extern bias_velocity

extern activation_buffer
extern delta_buffer 

extern output_buffer

extern init_random
extern init_params

extern printf
extern exit

section .data
    input_tensor: dd 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0

    test_tensor: dd 1.0, 1.0

    target_tensor: dd 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0

    fmt_output: db "Output[%d] = %.4f", 10, 0

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

    sub rsp, 176

    call init_random

    mov rcx, [input_neurons]
    mov rdx, [hidden_neurons]
    mov r8, [hidden_layers]
    mov r9, [output_neurons]
    call init_params

    mov r14, rax                ; weight tensor
    mov r15, rdx                ; bias tensor

    lea rcx, [input_tensor]
    lea rdx, [target_tensor]
    mov r8, [batch_size]
    mov r9, [input_neurons]

    mov rax, [hidden_neurons]
    mov [rsp + 32], rax

    mov rax, [hidden_layers]
    mov [rsp + 40], rax
    mov rax, [output_neurons]
    mov [rsp + 48], rax

    mov [rsp + 56], r14
    mov [rsp + 64], r15
    lea rax, [weight_grad_base_ptr]
    mov [rsp + 72], rax
    lea rax, [bias_grad_base_ptr]
    mov [rsp + 80], rax
    lea rax, [activation_buffer]
    mov [rsp + 88], rax
    lea rax, [delta_buffer]
    mov [rsp + 96], rax
    mov rax, [epochs]
    mov [rsp + 104], rax
    movsd xmm0, [learning_rate]
    movsd [rsp + 112], xmm0
    mov rax, [enable_dropout]
    mov [rsp + 120], rax
    movsd xmm0, [dropout_rate]
    movsd [rsp + 128], xmm0
    lea rax, [weight_velocity]
    mov [rsp + 136], rax
    lea rax, [bias_velocity]
    mov [rsp + 144], rax
    movsd xmm0, [momentum]
    movsd [rsp + 152], xmm0
    call mlp_train

    mov r14, rax                ; trained weights
    mov r15, rdx                ; trained biases

    lea rcx, [test_tensor]
    mov rdx, r14
    mov r8, r15
    lea r9, [activation_buffer]

    mov rax, [batch_size]
    mov [rsp + 32], rax
    mov rax, [input_neurons]
    mov [rsp + 40], rax
    mov rax, [hidden_neurons]
    mov [rsp + 48], rax
    mov rax, [hidden_layers]
    mov [rsp + 56], rax
    mov rax, [output_neurons]
    mov [rsp + 64], rax
    mov qword [rsp + 72], 0     ; disable dropout
    movsd xmm0, [dropout_rate]
    movsd [rsp + 80], xmm0
    call mlp_feed_forward

    mov ecx, [output_neurons]
    mov rsi, rax
    lea rdi, [output_buffer]
    rep movsd

    xor r12, r12
    mov rbx, [output_neurons]

.print_loop:
    cmp r12, rbx
    jge .done

    lea rcx, [fmt_output]
    mov rdx, r12
    lea rax, [output_buffer]
    movss xmm2, [rax + r12*4]
    cvtss2sd xmm2, xmm2
    movq r8, xmm2
    call printf

    inc r12
    jmp .print_loop 

.done:
    add rsp, 176

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rsi
    pop rbp

    xor ecx, ecx
    call exit
