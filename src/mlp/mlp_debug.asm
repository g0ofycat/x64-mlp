default rel

; @extern: C Libs

extern printf

; @section: String Data
section .data
    fmt_dbg: db "[rbp+%d]: Double: %lld | Float: %f | Address: 0x%p", 10, 0

; @section: Global labels
section .text
    global print_stack_offsets

; =============== PUBLIC LABELS ===============

; =============== print_stack_offsets ===============

; @function print_stack_offsets: Print all registers (inc. 8)
; @param: rcx - Caller Stack Frame
; @param: rdx - Offset Start
; @param: r8 - Offset Stop
print_stack_offsets:
    push rbx
    push r12
    push r13
    push r14

    sub rsp, 40

    mov r12, rcx
    mov r13, rdx
    mov r14, r8

    mov rbx, rdx

; @function .loop_start: Print stack offsets
.loop_start:
    lea rcx, [fmt_dbg]
    mov rdx, rbx
    mov r8, [r12 + rbx]
    movsd xmm3, [r12 + rbx]
    movq r9, xmm3
    lea rax, [r12 + rbx]
    mov [rsp + 32], rax
    call printf

    add rbx, 8
    cmp rbx, r14
    jl .loop_start

    add rsp, 40

    pop r14
    pop r13
    pop r12
    pop rbx

    ret
