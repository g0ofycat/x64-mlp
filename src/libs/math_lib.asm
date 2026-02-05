default rel

extern expf
extern logf

; @section: Global labels
section .text
    global relu
    global softmax
    global cross_entropy_loss

; =============== PUBLIC LABELS ===============

; =============== relu ===============

; @function relu: ReLU activation function (In Place)
; @param: rcx - The array to apply ReLU to
; @param: r8 - Array Length
relu:
    push rbx

    mov rbx, r8
    xor r10, r10
    xorps xmm1, xmm1

; @function .loop: Loop to calculate ReLU
.loop:
    cmp r10, rbx
    jge .done

    movss xmm0, [rcx + r10*4]
    maxss xmm0, xmm1
    movss [rcx + r10*4], xmm0

    inc r10

    jmp .loop

; @function .done: Label when ReLU loop is done
.done:
    pop rbx
    ret

; =============== softmax ===============

; @function softmax: Softmax activaton function
; @param: rcx - Pointer to input array
; @return rdx - Output to the softmaxxed array
softmax:
    push rbx
    push r12
    push r13
    push r14

    sub rsp, 32

    mov r12, rcx
    mov r13, rdx
    mov rbx, r8

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
    jge .compute_softmax

    ; exp(input[i] - max)
    movss xmm0, [r12 + r14*4]
    subss xmm0, xmm6           ; x - max
    call expf                  ; xmm0 = exp(xmm0)

    movss [r13 + r14*4], xmm0  ; store exp result
    addss xmm7, xmm0           ; sum += exp

    inc r14
    jmp .exp_loop

; @function .compute_softmax: xor to div by sum
.compute_softmax:
    xor r14, r14

; @function .divide_loop: Divide to get the average
.divide_loop:
    cmp r14, rbx
    jge .done

    movss xmm0, [r13 + r14*4]
    divss xmm0, xmm7
    movss [r13 + r14*4], xmm0

    inc r14
    jmp .divide_loop

; @function .done: Softmax done
.done:
    add rsp, 32

    pop r14
    pop r13
    pop r12
    pop rbx

    ret

; =============== cross_entropy_loss ===============

; @function cross_entropy_loss: Cross Entropy Loss Function
; @param: rcx - Pointer to prediction array
; @param: rdx - Pointer to label array
; @return: xmm0 - Loss Value 
cross_entropy_loss:
    push rbx
    push r12
    push r13
    push r14

    sub rsp, 32

    mov r12, rcx
    mov r13, rdx
    mov rbx, r8

    xorps xmm6, xmm6
    xor r14, r14                ; i = 0

; @function .loop: Loop to compute Cross Entropy Loss
.loop:
    cmp r14, rbx
    jge .done

    ; loss -= label * log(prediction)
    movss xmm0, [r12 + r14*4]   ; prediction
    call logf                   ; xmm0 = log(prediction)

    movss xmm1, [r13 + r14*4]   ; label
    mulss xmm0, xmm1            ; label * log(pred)
    subss xmm6, xmm0            ; sum -= result

    inc r14
    jmp .loop

; @function .done: Cross Entropy Loss Done
.done:
    movss xmm0, xmm6
 
    add rsp, 32
    pop r14
    pop r13
    pop r12
    pop rbx

    ret
