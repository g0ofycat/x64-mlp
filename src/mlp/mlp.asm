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
; @param: [rsp+56] - Input rows (batch size)
; @param: [rsp+64] - Input columns (input neurons)
; @param: [rsp+72] - Output columns (output neurons)
; @param: [rsp+80] - Apply dropout (0 or 1)
; @param: [rsp+88] - If dropout, dropout rate
; @return: rax - Pointer to output matrix
mlp_feed_forward:
    push rbp

    mov rbp, rsp

    push rbx
    push r12
    push r13
    push r14
    push r15

    sub rsp, 32

    mov r12, rcx            ; input pointer
    mov r13, rdx            ; weight pointer
    mov r14, r8             ; bias pointer
    mov r15, r9             ; output pointer

    mov rbx, [rbp + 56]     ; rows (batch size)
    mov r10, [rbp + 64]     ; input_cols
    mov r11, [rbp + 72]     ; output_cols

    xor r8, r8              ; r = 0

; @function .row_loop: Apply activation to each row
.row_loop:
    cmp r8, rbx
    jge .apply_activation

    xor r9, r9

; @function .col_loop: Iterate through each col
.col_loop:
    cmp r9, r11
    jge .next_row

    movss xmm6, [r14 + r9*4] ; bias[j]

    xor rax, rax             ; k = 0

; @function .inner_loop: Calculate output after forward pass
.inner_loop:
    cmp rax, r10
    jge .store_output

    ; output[i,j] += input[i,k] * weight[k,j]

    ; calc input[i,k]

    mov rcx, r8
    imul rcx, r10               ; rax = i * input_cols

    add rcx, rax                ; rax = i * input_cols + k
    movss xmm0, [r12 + rcx*4]   ; xmm0 = input[i,k]

    ; calc weight[k,j]
    mov rdx, rax
    imul rdx, r11               ; rax = k * output_cols
    add rdx, r9                 ; rax = k * output_cols + j

    movss xmm1, [r13 + rdx*4]   ; xmm1 = weight[k,j]

    ; mult and accum
    mulss xmm0, xmm1
    addss xmm6, xmm0

    inc rax
    jmp .inner_loop

; @function .store_output: Store output and jump to next col
.store_output:
    mov rax, r8

    imul rax, r11
    add rax, r9

    movss [r15 + rax*4], xmm6

    inc r9
    jmp .col_loop

; @function .next_row: Increment and jump to next row
.next_row:
    inc r8
    jmp .row_loop

; @function .apply_activation: Apply activation function
.apply_activation:
    mov rcx, r15            ; output pointer
    mov rdx, r15
    mov rax, rbx
    imul rax, r11           ; total elements
    mov r8, rax             ; len
    call relu
    
    cmp qword [rsp + 80], 0
    je .done

    mov rcx, r15
    mov rax, rbx

    imul rax, r11

    mov r8, rax

    movss xmm0, [rsp + 88]
    call apply_dropout

; @function .done: When Feed Forward is done
.done:
    mov rax, r15

    add rsp, 32

    pop r15
    pop r14
    pop r13
    pop r12
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
