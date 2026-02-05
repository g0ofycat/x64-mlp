default rel

; @extern: C Libs

extern rand
extern srand
extern time

; @section: Data for apply_dropout
section .data
    one dd 1.0
    rand_max_float dd 32767.0       ; at least on cpp im pretty sure this is the lowest constant
    min_keep dd 0.00001

; @section: Global labels
section .text
    global init_random
    global apply_dropout

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

    sub rsp, 56

    mov r12, rcx        ; array pointer
    mov r13, r8         ; size

    movss xmm2, [one]
    subss xmm2, xmm0    ; keep_prob = 1 - dropout_rate
    maxss xmm2, [min_keep]

    xor rbx, rbx        ; i = 0

; @function .loop: Loop to apply dropout
.loop:
    cmp rbx, r13
    jge .done

    call rand               ; eax = random int
    cvtsi2ss xmm0, eax      ; eax to float
    movss xmm1, [rand_max_float]
    divss xmm0, xmm1

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
    xorps xmm0, xmm0
    movss [r12 + rbx*4], xmm0

; @function .next: Jump to the next neuron
.next:
    inc rbx
    jmp .loop

; @function .done: Label when dropout is done
.done:
    add rsp, 56

    pop r13
    pop r12
    pop rbx

    ret
