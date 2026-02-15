default rel

extern expf
extern logf

; @section: CEL Data
section .data
    cel_epsilon: dd 0x358637BD      ; 1e-6 f32

; @section: Global labels
section .text
    global relu
    global relu_derivative
    global softmax
    global cross_entropy_loss

; =============== PUBLIC LABELS ===============

; =============== relu ===============

; @function relu: ReLU activation function (in-place)
; @param: rcx - The array to apply ReLU to
; @param: r8 - Array Length
relu:
    xor r10, r10
    xorps xmm1, xmm1

    mov r9, r8
    and r9, -4

; @function .loop: Loop to calculate ReLU (4 SIMD)
.simd_loop:
    cmp r10, r9
    jge .scalar_tail

    movups xmm0, [rcx + r10*4]
    maxps xmm0, xmm1
    movups [rcx + r10*4], xmm0

    add r10, 4
    jmp .simd_loop

; @function .scalar_tail: Calculate remaining ReLU if not a multiple of 4
.scalar_tail:
    cmp r10, r8
    jge .done

    movss xmm0, [rcx + r10*4]
    maxss xmm0, xmm1
    movss [rcx + r10*4], xmm0

    inc r10
    jmp .scalar_tail

; @function .done: Label when relu is done
.done:
    ret

; =============== relu_derivative ===============

; @function relu_derivative: ReLU Derivative (in-place)
; @param: rcx - Pointer to original output (forward pass)
; @param: rdx - Pointer to gradients (to be modified in-place)
; @param: r8 - Array Length
relu_derivative:
    xor r10, r10
    xorps xmm2, xmm2

    mov r9, r8
    and r9, -4

; @function .simd_loop: Loop to calculate ReLU Derivative (4 SIMD)
.simd_loop:
    cmp r10, r9
    jge .scalar_tail

    movups xmm0, [rcx + r10*4]
    movups xmm1, [rdx + r10*4]

    movaps xmm3, xmm2
    cmpltps xmm3, xmm0

    andps xmm1, xmm3
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

    andps xmm1, xmm3
    movss [rdx + r10*4], xmm1

    inc r10
    jmp .scalar_tail

; @function .done: Label when relu_derivative is done
.done:
    ret

; =============== softmax ===============

; @function softmax: Softmax activaton function
; @param: rcx - Pointer to input array
; @param: rdx - Pointer to output array
; @param: r8 - Array length
softmax:
    push rbx
    push r12
    push r13
    push r14

    sub rsp, 88     ; todo: xmm6 xmm7 push stack (?)

    mov r12, rcx
    mov r13, rdx
    mov rbx, r8

    cmp rbx, 1
    jle .single_element

    movss xmm6, [r12]
    xor r14, r14               ; i = 0

; @function .find_max_loop: Find the max number in rcx
.find_max_loop:
    cmp r14, rbx
    jge .found_max

    movss xmm0, [r12 + r14*4]
    maxss xmm6, xmm0

    inc r14

    jmp .find_max_loop

; @function .found_max: xor counter and xmm7
.found_max:
    xorps xmm7, xmm7
    xor r14, r14

; @function .exp_loop: Loop to find exp sum
.exp_loop:
    cmp r14, rbx
    jge .tensor_divide

    ; exp(input[i] - max)
    movss xmm0, [r12 + r14*4]
    subss xmm0, xmm6           ; x - max
    call expf                  ; xmm0 = exp(xmm0)

    movss [r13 + r14*4], xmm0  ; store exp result
    addss xmm7, xmm0           ; sum += exp

    inc r14
    jmp .exp_loop

; @function .tensor_divide: SIMD Average calculate (4 SIMD)
.tensor_divide:
    shufps xmm7, xmm7, 0
    xor r14, r14

    mov rax, rbx
    and rax, -4

; @function .div_simd: Divide SIMD
.div_simd:
    cmp r14, rax
    jge .div_scalar

    movups xmm0, [r13 + r14*4]
    divps xmm0, xmm7
    movups [r13 + r14*4], xmm0

    add r14, 4
    jmp .div_simd

; @function .div_scalar: Divide remaining scalars
.div_scalar:
    cmp r14, rbx
    jge .done

    movss xmm0, [r13 + r14*4]
    divss xmm0, xmm7
    movss [r13 + r14*4], xmm0

    inc r14
    jmp .div_scalar

; @function .single_element: 1 logit, skip softmax and copy to output array
.single_element:
    movss xmm0, [r12]
    movss [r13], xmm0

; @function .done: Label when softmax is done
.done:
    add rsp, 88

    pop r14
    pop r13
    pop r12
    pop rbx

    ret

; =============== cross_entropy_loss ===============

; @function cross_entropy_loss: Cross Entropy Loss Function
; @param: rcx - Pointer to prediction array
; @param: rdx - Pointer to label array
; @param: r8 - Array length
; @return: xmm0 - Loss Value 
cross_entropy_loss:
    push rbx
    push r12
    push r13
    push r14

    sub rsp, 40

    mov r12, rcx
    mov r13, rdx
    mov rbx, r8

    movss xmm7, [cel_epsilon]

    xorps xmm6, xmm6
    xor r14, r14                ; i = 0

; @function .loop: Loop to compute Cross Entropy Loss
.loop:
    cmp r14, rbx
    jge .done

    movss xmm0, [r12 + r14*4]   ; prediction
    maxss xmm0, xmm7
    call logf                   ; xmm0 = log(prediction)

    mulss xmm0, [r13 + r14*4]   ; label * log(pred)
    subss xmm6, xmm0            ; loss -= label * log(pred)

    inc r14
    jmp .loop

; @function .done: Label when cross_entropy_loss is done
.done:
    movss xmm0, xmm6

    add rsp, 40

    pop r14
    pop r13
    pop r12
    pop rbx

    ret
