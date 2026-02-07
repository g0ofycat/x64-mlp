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
extern init_params

extern printf
extern exit

section .data
    input_matrix dd 0.5, 0.8, 0.3, 0.9

    fmt_cycles db "CPU Cycles: %d", 10, 0
    fmt_output db "Output[%d] = %.4f", 10, 0

section .bss
    output_matrix resd 3

section .text
    global main

main:
    sub rsp, 72

    call init_random

    xor eax, eax
    cpuid
    rdtsc
    shl rdx, 32
    or  rax, rdx
    mov r13, rax 

    mov rcx, [input_neurons]
    mov rdx, [hidden_neurons]
    mov r8, [hidden_layers]
    mov r9, [output_neurons]
    call init_params

    mov r10, rax
    mov r11, rdx

    lea rcx, [input_matrix]
    mov rdx, r10
    mov r8, r11
    lea r9, [output_matrix]

    mov qword [rsp + 40], 1
    mov qword [rsp + 48], 4
    mov qword [rsp + 56], 3
    mov qword [rsp + 64], 0
    mov qword [rsp + 72], 0

    call mlp_feed_forward

    rdtscp
    shl rdx, 32
    or  rax, rdx
    mov rax, r12

    cpuid
    sub r13, r12

    lea rcx, [fmt_cycles]
    mov rdx, r13
    call printf

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
