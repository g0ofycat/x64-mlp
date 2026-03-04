default rel

extern expf
extern logf

; @section: CEL Data
section .data
    leaky_alpha: dd 0.01
    leaky_alpha_ps: dd 0.01, 0.01, 0.01, 0.01

    cel_epsilon: dd 0x358637BD      ; 1e-6 f32

; @section: Global labels
section .text
    global leaky_relu
    global leaky_relu_derivative

    global softmax
    global cross_entropy_loss

; =============== PUBLIC LABELS ===============

; =============== relu ===============

; @function leaky_relu: Leaky ReLU activation function (in-place)
; @param: rcx - The array to apply ReLU to
; @param: r8 - Array Length
leaky_relu:
    xor r10, r10
    movups xmm2, [leaky_alpha_ps]
    xorps xmm3, xmm3

    mov r9, r8
    and r9, -4

; @function .loop: Loop to calculate ReLU (4 SIMD)
.simd_loop:
    cmp r10, r9
    jge .scalar_tail

    movups xmm0, [rcx + r10*4]
    movaps xmm1, xmm0
    mulps xmm1, xmm2            ; x * 0.01
    maxps xmm0, xmm3            ; max(x, 0)
    movaps xmm4, xmm3
    cmpltps xmm4, xmm0
    andps xmm0, xmm4
    andnps xmm4, xmm1
    orps xmm0, xmm4
    movups [rcx + r10*4], xmm0

    add r10, 4
    jmp .simd_loop

; @function .scalar_tail: Calculate remaining ReLU if not a multiple of 4
.scalar_tail:
    cmp r10, r8
    jge .done

    movss xmm0, [rcx + r10*4]
    movaps xmm1, xmm0
    mulss xmm1, [leaky_alpha]   ; x * 0.01
    maxss xmm0, xmm3            ; max(x, 0)
    movaps xmm4, xmm3
    cmpltss xmm4, xmm0
    andps xmm0, xmm4
    andnps xmm4, xmm1
    orps xmm0, xmm4
    movss [rcx + r10*4], xmm0

    inc r10
    jmp .scalar_tail

; @function .done: Label when relu is done
.done:
    ret

; =============== relu_derivative ===============

; @function leaky_relu_derivative: Leaky ReLU Derivative (in-place)
; @param: rcx - Pointer to original output (forward pass)
; @param: rdx - Pointer to gradients (to be modified in-place)
; @param: r8 - Array Length
leaky_relu_derivative:
    xor r10, r10
    xorps xmm2, xmm2
    movups xmm5, [leaky_alpha_ps]

    mov r9, r8
    and r9, -4

; @function .simd_loop: Loop to calculate ReLU Derivative (4 SIMD)
.simd_loop:
    cmp r10, r9
    jge .scalar_tail

    movups xmm0, [rcx + r10*4]  ; activations
    movups xmm1, [rdx + r10*4]  ; gradients

    movaps xmm3, xmm2
    cmpltps xmm3, xmm0

    movaps xmm4, xmm5
    mulps xmm4, xmm1

    andps xmm1, xmm3
    andnps xmm3, xmm4
    orps xmm1, xmm3
    movups [rdx + r10*4], xmm1

    add r10, 4
    jmp .simd_loop

; @function .scalar_tail: Calculate remaining ReLU Derivatives if not a multiple of 4
.scalar_tail:
    cmp r10, r8
    jge .done

    movss xmm0, [rcx + r10*4]
    movss xmm1, [rdx + r10*4]

    pxor xmm3, xmm3
    cmpltss xmm3, xmm0

    movaps xmm4, xmm5
    mulss xmm4, xmm1
    andps xmm1, xmm3
    andnps xmm3, xmm4
    orps xmm1, xmm3
    movss [rdx + r10*4], xmm1

    inc r10
    jmp .scalar_tail

; @function .done: Label when relu_derivative is done
.done:
    ret

; =============== softmax ===============

; @function softmax: Softmax activation function
; @param: rcx - Pointer to input array
; @param: rdx - Pointer to output array
; @param: r8 - Array length
softmax:
    push rbp
    mov rbp, rsp

    push rbx
    push r12
    push r13
    push r14

    sub rsp, 80

    mov r12, rcx
    mov r13, rdx
    mov rbx, r8

    cmp rbx, 1
    jle .single_element

    movss xmm4, [r12]          ; max
    xor r14, r14               ; i = 0

; @function .find_max_loop: Find the max number in rcx
.find_max_loop:
    cmp r14, rbx
    jge .found_max

    movss xmm0, [r12 + r14*4]
    maxss xmm4, xmm0

    inc r14
    jmp .find_max_loop

; @function .found_max: xor counter and clear sum
.found_max:
    xorps xmm5, xmm5           ; sum
    xor r14, r14

; @function .exp_loop: Loop to find exp sum
.exp_loop:
    cmp r14, rbx
    jge .tensor_divide

    movss xmm0, [r12 + r14*4]
    subss xmm0, xmm4           ; x - max

    movss [rsp + 40], xmm4
    movss [rsp + 48], xmm5

    call expf                  ; exp(xmm0)

    movss xmm4, [rsp + 40]
    movss xmm5, [rsp + 48]

    movss [r13 + r14*4], xmm0
    addss xmm5, xmm0           ; sum += exp

    inc r14
    jmp .exp_loop

; @function .tensor_divide: SIMD divide (4 SIMD)
.tensor_divide:
    shufps xmm5, xmm5, 0
    xor r14, r14

    mov rax, rbx
    and rax, -4

; @function .div_simd: Divide SIMD
.div_simd:
    cmp r14, rax
    jge .div_scalar

    movups xmm0, [r13 + r14*4]
    divps xmm0, xmm5
    movups [r13 + r14*4], xmm0

    add r14, 4
    jmp .div_simd

; @function .div_scalar: Divide remaining scalars
.div_scalar:
    cmp r14, rbx
    jge .done

    movss xmm0, [r13 + r14*4]
    divss xmm0, xmm5
    movss [r13 + r14*4], xmm0

    inc r14
    jmp .div_scalar

; @function .single_element: 1 logit, skip softmax and copy to output array
.single_element:
    movss xmm0, [r12]
    movss [r13], xmm0

; @function .done: Label when softmax is done
.done:
    add rsp, 80

    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp

    ret