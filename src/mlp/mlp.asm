default rel

; @extern: Math Functions

extern relu
extern softmax
extern cross_entropy_loss

; @extern: External Libraries

extern apply_dropout

; @section: Global labels
section .text
    global mlp_feed_forward
    global mlp_train
    global mlp_back_propagation

; =============== PUBLIC LABELS ===============

; =============== mlp_feed_forward ===============

; @function mlp_feed_forward: Feed forward pass through one layer
; @param: rcx - Input matrix pointer
; @param: rdx - Weight matrix pointer
; @param: r8  - Bias vector pointer
; @param: r9  - Output matrix pointer (pre-allocated)
; @param: [rbp+40] - Input rows (batch size)
; @param: [rbp+48] - Input columns (input neurons)
; @param: [rbp+56] - Output columns (output neurons)
; @param: [rbp+64] - Apply dropout (0 or 1)
; @param: [rbp+72] - If dropout, dropout rate
; @return: rax - Pointer to output matrix
mlp_feed_forward:
    push rbp

    mov rbp, rsp

    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15

    sub rsp, 88

    mov r12, rcx            ; input pointer
    mov r13, rdx            ; weight pointer
    mov r14, r8             ; bias pointer
    mov r15, r9             ; output pointer

    mov rbx, [rbp + 56]      ; input_rows (batch size)
    mov r10, [rbp + 64]      ; input_cols
    mov r11, [rbp + 72]      ; output_cols

    xor rsi, rsi              ; i = 0

; @function .row_loop: Apply activation to each row
.row_loop:
    cmp rsi, rbx
    jge .apply_activation

    mov rax, rsi

    imul rax, r11
    lea rdi, [r15, rax*4]

    xor r8, r8                ; j = 0

; @function .bias_fill: Fill bias based on output columns
.bias_fill:
    cmp r8, r11
    jge .start_k_loop

    movss xmm0, [r14 + r8*4]
    movss [rdi + r8*4], xmm0

    inc r8
    jmp .bias_fill

; @function .start_k_loop: xor rcx
.start_k_loop:
    xor rcx, rcx              ; k = 0

; @function .k_loop: Loop through current column
.k_loop:
    cmp rcx, r10
    jge .next_row

    mov rax, rsi
    imul rax, r10
    add rax, rcx

    movss xmm0, [r12 + rax*4]
    shufps xmm0, xmm0, 0

    mov rax, rcx
    imul rax, r11
    lea rdx, [r13 + rax*4]

    xor r8, r8               ; j = 0

; @function .j_loop_simd: Loop through current row
.j_loop_simd:
    mov rax, r11
    sub rax, r8

    cmp rax, 4
    jl .j_loop_scalar

    movups xmm1, [rdx + r8*4]
    movups xmm2, [rdi + r8*4]

    mulps xmm1, xmm0
    addps xmm2, xmm1

    movups [rdi + r8*4], xmm2

    add r8, 4
    jmp .j_loop_simd

; @function .j_loop_scalar: Scalar mult
.j_loop_scalar:
    cmp r8, r11
    jge .next_k

    movss xmm1, [rdx + r8*4]
    mulss xmm1, xmm0
    addss xmm1, [rdi + r8*4]
    movss [rdi + r8*4], xmm1

    inc r8
    jmp .j_loop_scalar

; @function .next_k: Jump to the next column
.next_k:
    inc rcx
    jmp .k_loop

; @function .next_row: Jump to the next row
.next_row:
    inc rsi
    jmp .row_loop

; @function .apply_activation: Apply activation function
.apply_activation:
    mov rcx, r15            ; output pointer
    mov rax, rbx
    imul rax, r11           ; total elements
    mov r8, rax             ; len

    call relu

    cmp qword [rbp + 80], 0
    je .do_softmax

    mov rcx, r15
    mov rax, rbx

    imul rax, r11

    mov r8, rax

    movss xmm0, [rbp + 88]

    call apply_dropout

; @function .do_softmax: Do softmax on r15
.do_softmax:
    mov rcx, r15
    mov rdx, r15

    mov r8, r11

    call softmax

; @function .done: When Feed Forward is done
.done:
    mov rax, r15

    add rsp, 88

    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp

    ret

; =============== mlp_train ===============

; @function mlp_train: Train the MLP using Inputs and Targets
; @param: rcx - Source to the input matrix
; @param: r9 - Source to the target matrix
; @return: rax - Pointer to the weight matrix
; @return: rax[rsp+8] - Pointer to the bias vector
mlp_train:

; =============== mlp_back_propagation ===============

; @function mlp_back_propagation: Used by mlp_train for back propagation
; @param: rcx - Source to the input matrix
; @param: r9 - Source to the target matrix
; @return: rax - Pointer to the updated weight matrix
; @return: rax[rsp+8] - Pointer to the updated bias vector
mlp_back_propagation:
