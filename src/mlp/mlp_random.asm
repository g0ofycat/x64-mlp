default rel

; @extern: C Libs

extern rand
extern srand
extern time
extern malloc

; @section: Data for apply_dropout
section .data
    one_f: dd 1.0
    two_f: dd 2.0
    six_f: dd 6.0

    rand_max_float: dd 32767.0       ; at least on cpp im pretty sure this is the lowest constant
    min_keep: dd 0.00001

; @section: Global labels
section .text
    global init_random
    global apply_dropout
    global init_params

; =============== PUBLIC LABELS ===============

; =============== init_random ===============

; @function init_random: Init random seed
init_random:
    sub rsp, 40

    xor rcx, rcx

    call time

    mov rcx, rax
    call srand

    add rsp, 40

    ret

; =============== apply_dropout ===============

; @function apply_dropout: Apply dropout to array (in-place)
; @param: rcx - Pointer to the array
; @param: r8 - Size of the array
; @param: xmm0 - Dropout Rate
apply_dropout:
    push rbx
    push r12
    push r13

    sub rsp, 64

    mov r12, rcx        ; array pointer
    mov r13, r8         ; size

    movss xmm2, [one_f]
    subss xmm2, xmm0    ; keep_prob = 1 - dropout_rate
    maxss xmm2, [min_keep]
    movss [rsp + 40], xmm2

    xor rbx, rbx        ; i = 0

; @function .loop: Loop to apply dropout
.loop:
    cmp rbx, r13
    jge .done

    call rand               ; eax = random int

    and eax, 0x7FFF
    cvtsi2ss xmm0, eax      ; eax to float
    divss xmm0, [rand_max_float]

    movss xmm2, [rsp + 40]

    comiss xmm0, xmm2
    ja .drop

; @function .keep: Keep current neuron
.keep:
    movss xmm0, [r12 + rbx*4]
    divss xmm0, xmm2

    movss [r12 + rbx*4], xmm0
    jmp .next

; @function .drop: Drop current neuron
.drop:
    pxor xmm0, xmm0
    movss [r12 + rbx*4], xmm0

; @function .next: Jump to the next neuron
.next:
    inc rbx
    jmp .loop

; @function .done: Label when apply_dropout is done
.done:
    add rsp, 64

    pop r13
    pop r12
    pop rbx

    ret

; =============== init_params ===============

; @function init_params: Init weights (He Uniform) and biases (0)
; @param rcx - Amount of input neurons
; @param rdx - Amount of hidden neurons
; @param r8 - Amount of hidden layers
; @param r9 - Amount of output neurons
; @return rax - Pointer to weight tensor
; @return rdx - Pointer to the bias tensor
init_params:
    push rbp
    mov rbp, rsp

    push r12
    push r13
    push r14
    push r15
    push rdi
    push rsi

    sub rsp, 80

    mov [rbp - 8], rcx
    mov [rbp - 16], rdx
    mov [rbp - 24], r8
    mov [rbp - 32], r9

    mov rax, [rbp - 8]
    imul rax, [rbp - 16]

    mov r12, rax

    mov rsi, [rbp - 24]
    dec rsi
    jz .skip_hidden_calc
    
    mov rax, [rbp - 16]
    imul rax, [rbp - 16]
    imul rax, rsi

    add r12, rax

; @function .skip_hidden_calc: Skip if only 1 hidden layer exists
.skip_hidden_calc:
    mov rax, [rbp - 16]
    imul rax, [rbp - 32]

    add r12, rax

    mov rax, [rbp - 16]
    imul rax, [rbp - 24]
    add rax, [rbp - 32]

    mov r13, rax

    mov rcx, r12
    shl rcx, 2
    call malloc
    mov r14, rax

    mov rcx, r13
    shl rcx, 2
    call malloc
    mov r15, rax

    mov rdi, r14

    mov rcx, rdi

    mov rdx, [rbp - 8]
    imul rdx, [rbp - 16]

    mov rax, [rbp - 8]

    call .he_fill

    mov rax, [rbp - 8]
    imul rax, [rbp - 16]

    shl rax, 2
    add rdi, rax

    mov rsi, 1

; @function .hidden_loop: Loop through hidden layers for weights
.hidden_loop:
    mov r10, [rbp - 24]
    cmp rsi, r10
    jge .final_layer

    mov rax, [rbp - 16]
    mov rcx, rdi

    mov rdx, [rbp - 16]
    imul rdx, [rbp - 16]

    call .he_fill

    mov rax, [rbp - 16]
    imul rax, [rbp - 16]

    shl rax, 2
    add rdi, rax

    inc rsi
    jmp .hidden_loop

; @function .final_layer: Initialize final weights
.final_layer:
    mov rax, [rbp - 16]
    mov rcx, rdi

    mov rdx, [rbp - 16]
    imul rdx, [rbp - 32]
    call .he_fill

    mov rdi, r15
    xor eax, eax
    mov rcx, r13
    rep stosd

    mov rax, r14
    mov rdx, r15

    add rsp, 80

    pop rsi
    pop rdi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp

    ret

; =============== INTERNAL HELPERS ===============

; @function .he_fill: Fill a buffer with He Uniform distribution (in-place)
; @param rcx - Buffer pointer
; @param rdx - Element count
; @param rax - fan_in (for limit calculation)
.he_fill:
    push rbx
    push r12
    push r13
    push r14

    sub rsp, 56

    mov rbx, rcx
    mov r12, rdx
    mov r13, rax

    cvtsi2ss xmm0, r13
    movss xmm1, [six_f]
    divss xmm1, xmm0
    sqrtss xmm0, xmm1 
    movss [rsp + 40], xmm0

    xor r14, r14

; @function .f_loop: Loop to generate random floats
.f_loop:
    cmp r14, r12
    jge .f_done

    call rand

    and eax, 0x7FFF
    cvtsi2ss xmm0, eax
    divss xmm0, [rand_max_float]
    mulss xmm0, [two_f]
    subss xmm0, [one_f]
    mulss xmm0, [rsp + 40]      ; todo: check if rand overrides xmm0

    movss [rbx + r14*4], xmm0

    inc r14
    jmp .f_loop

; @function .f_done: Cleanup and return from fill
.f_done:
    add rsp, 56

    pop r14
    pop r13
    pop r12
    pop rbx

    ret
