default rel

; @extern: C Libs

extern printf

; @section: String Data
section .data
    fmt_dbg: db "[rbp+%d]: Double: %lld | Float: %f | Address: 0x%p", 10, 0
    dbg_f32_fmt: db "[%lld]: %f", 10, 0

; @section: Global labels
section .text
    global print_stack_offsets
    global print_f32_array

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

; =============== print_f32_array ===============

; @function print_f32_array: Print f32 array contents
; @param: rcx - Array Pointer
; @param: rdx - Array Length
print_f32_array:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi
    push r12
    push r13

    sub rsp, 32

    mov r12, rcx                   ; array ptr
    mov r13, rdx                   ; length
    xor rsi, rsi                   ; i = 0

; @function .print_loop: Loop to print all f32's
.print_loop:
    cmp rsi, r13
    jge .done

    lea rcx, [dbg_f32_fmt]
    mov rdx, rsi
    movss xmm2, [r12 + rsi*4]
    cvtss2sd xmm2, xmm2
    movq r8, xmm2
    call printf

    inc rsi

    jmp .print_loop

; @function .done: Label when print_f32_array is done
.done:
    add rsp, 32

    pop r13
    pop r12
    pop rsi
    pop rbx
    pop rbp

    ret
