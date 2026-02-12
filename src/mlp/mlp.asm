default rel

; @extern: Math Functions

extern relu
extern relu_derivative
extern softmax
extern cross_entropy_loss

; @extern: External Libraries

extern apply_dropout

; @extern: C Libs

extern printf

; @section: String Data
section .data
    fmt_cel: db "Loss: %.6f Epoch: %lld", 10, 0

; @section: Global labels
section .text
    global mlp_feed_forward
    global mlp_train
    global mlp_back_propagation

    global mlp_forward_layer
    global mlp_backward_layer

; =============== PUBLIC LABELS ===============

; =============== mlp_feed_forward ===============

; @function mlp_feed_forward: Feed forward pass through the network
; @param: rcx - Input tensor pointer
; @param: rdx - Weight tensor pointer
; @param: r8 - Bias tensor pointer
; @param: r9 - Output tensor pointer (pre-allocated)
; @param: [rbp+40] - Input rows (batch size)
; @param: [rbp+48] - Input columns (input neurons)
; @param: [rbp+56] - Amount of hidden neurons per layer
; @param: [rbp+64] - Amount of hidden layers
; @param: [rbp+72] - Output columns (output neurons)
; @param: [rbp+80] - Apply dropout (0 or 1)
; @param: [rbp+88] - If dropout, dropout rate
; @return: rax - Pointer to output tensor
mlp_feed_forward:
    push rbp
    mov rbp, rsp

    push rbx
    push r12
    push r13
    push r14
    push r15

    sub rsp, 104

    mov r12, rcx                  ; input pointer
    mov r13, rdx                  ; weight pointer
    mov r14, r8                   ; bias pointer
    mov r15, r9                   ; output pointer

    mov rbx, [rbp + 72]           ; hidden layers

    xor rsi, rsi                  ; layer index = 0
    mov rdx, r12                  ; acts_prev = input

; @function .loop_layers: Check current layer index and see if first index
.loop_layers:
    cmp rsi, rbx
    jge .output_layer

    mov r8, [rbp + 56]            ; input neurons

    test rsi, rsi
    jne .not_first

    jmp .in_set

; @function .not_first: r8 = hidden neurons
.not_first:
    mov r8, [rbp + 64]            ; hidden neurons

; @function .in_set: r9 = hidden neurons
.in_set:
    mov r9, [rbp + 64]            ; hidden neurons

; @function .out_set: mlp_forward_layer loop
.out_set:
    mov rax, rsi
    imul rax, [rbp + 64]          ; layer * hidden_neurons
    imul rax, [rbp + 48]          ; * batch_size
    lea rax, [r15 + rax * 4]

    mov [rsp + 32], r8            ; in_neurons
    mov [rsp + 40], r9            ; out_neurons
    mov [rsp + 48], rax           ; acts_curr

    mov rcx, rdx                  ; acts_prev
    mov rdx, r13                  ; weights
    mov r8, r14                   ; biases
    mov r9, rax                   ; acts_curr

    mov rax, [rbp + 48]
    mov [rsp + 40], rax           ; batch size
    mov rax, [rsp + 32]
    mov [rsp + 48], rax           ; in_neurons
    mov rax, [rsp + 40]
    mov [rsp + 56], rax           ; out_neurons
    mov rax, [rbp + 88]
    mov [rsp + 64], rax           ; dropout enabled
    movss xmm0, [rbp + 96]
    movss [rsp + 72], xmm0        ; dropout_rate
    call mlp_forward_layer

    mov r8, [rsp + 32]            ; restore in_neurons
    mov r9, [rsp + 40]            ; restore out_neurons
    mov rax, r8
    imul rax, r9

    lea r13, [r13 + rax * 4]      ; next weight block
    lea r14, [r14 + r9 * 4]       ; next bias block

    mov rdx, rax                   ; acts_prev = acts_curr
    inc rsi
    jmp .loop_layers

; @function .output_layer: Softmax and return when mlp_feed_forward is done
.output_layer:
    mov rcx, rdx                  ; acts_prev (last hidden layer)
    mov rdx, r13                  ; weights (current position)
    mov r8, r14                   ; biases (current position)
    mov r9, r15                   ; output buffer

    mov rax, [rbp + 48]
    mov [rsp + 40], rax           ; batch size
    mov rax, [rbp + 64]
    mov [rsp + 48], rax           ; in_neurons
    mov rax, [rbp + 80]
    mov [rsp + 56], rax           ; out_neurons
    mov qword [rsp + 64], 0       ; dropout enabled
    movss xmm0, [rbp + 96]
    movss [rsp + 72], xmm0        ; dropout_rate
    call mlp_forward_layer

    mov rcx, r15                  ; input buffer
    mov rdx, r15                  ; output buffer
    mov r8, [rbp + 80]            ; output neurons
    call softmax

    mov rax, r15

    add rsp, 104

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp

    ret

; =============== mlp_train ===============

; @function mlp_train: Train MLP (SGD)
; @param: rcx - Input tensor pointer
; @param: rdx - Target tensor pointer
; @param: r8 - Input rows (batch size)
; @param: r9 - Input columns (input neurons)
; @param: [rbp+40] - Amount of hidden neurons per layer
; @param: [rbp+48] - Amount of hidden layers
; @param: [rbp+56] - Output columns (output neurons)
; @param: [rbp+64] - Weight tensor pointer
; @param: [rbp+72] - Bias tensor pointer
; @param: [rbp+80] - Weight gradient buffer
; @param: [rbp+88] - Bias gradient buffer
; @param: [rbp+96] - Activation buffer (flat)
; @param: [rbp+104] - Delta buffer (flat)
; @param: [rbp+112] - Amount of epochs
; @param: [rbp+120] - Learning rate
; @param: [rbp+128] - Apply dropout (0 or 1)
; @param: [rbp+136] - If dropout, dropout rate
; @return: rax - Weight tensor pointer
; @return: rdx - Bias tensor pointer
mlp_train:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi
    push r12
    push r13
    push r14
    push r15

    sub rsp, 144

    mov r12, r8                     ; batch_size
    mov r13, r9                     ; input_neurons
    mov r14, rcx                    ; input tensor
    mov r15, rdx                    ; target tensor

    mov rbx, [rbp + 248]            ; epochs

    test rbx, rbx
    jz .done

; @function .epoch_loop: Epoch training loop
.epoch_loop:
    mov rcx, r14                    ; input tensor

    mov rdx, [rbp + 200]            ; weights
    mov r8, [rbp + 208]             ; biases
    mov r9, [rbp + 232]             ; activations

    mov rax, r12
    mov [rsp + 48], rax             ; batch_size
    mov rax, r13
    mov [rsp + 56], rax             ; input_neurons
    mov rax, [rbp + 176]
    mov [rsp + 64], rax             ; hidden_neurons
    mov rax, [rbp + 184]
    mov [rsp + 72], rax             ; num_hidden
    mov rax, [rbp + 192]
    mov [rsp + 80], rax             ; output_neurons
    mov rax, [rbp + 264]
    mov [rsp + 88], rax             ; enable dropout
    movss xmm0, [rbp + 272]
    movss [rsp + 96], xmm0          ; dropout_rate
    call mlp_feed_forward

    mov [rbp - 8], rax              ; save predictions

    mov rcx, rax                    ; predictions
    mov rdx, r15                    ; targets
    mov r8, [rbp + 192]             ; output_neurons (per sample)
    imul r8, r12                    ; * batch_size = total length
    call cross_entropy_loss

    lea rcx, [fmt_cel]
    movq rdx, xmm0
    mov r8, rbx
    call printf

    mov rcx, r14                    ; input tensor
    mov rdx, r15                    ; target tensor
    mov r8, r12                     ; batch_size
    mov r9, r13                     ; input_neurons

    mov rax, [rbp - 8]
    mov [rsp + 32], rax             ; predictions
    mov rax, [rbp + 176]
    mov [rsp + 40], rax             ; hidden_neurons
    mov rax, [rbp + 184]
    mov [rsp + 48], rax             ; num_hidden
    mov rax, [rbp + 192]
    mov [rsp + 56], rax             ; output_neurons
    mov rax, [rbp + 200]
    mov [rsp + 64], rax             ; weights
    mov rax, [rbp + 208]
    mov [rsp + 72], rax             ; biases
    mov rax, [rbp + 216]
    mov [rsp + 80], rax             ; weight_grads
    mov rax, [rbp + 224]
    mov [rsp + 88], rax             ; bias_grads
    mov rax, [rbp + 232]
    mov [rsp + 96], rax             ; activations
    mov rax, [rbp + 240]
    mov [rsp + 104], rax            ; deltas
    call mlp_back_propagation

    xor rsi, rsi                    ; layer = 0

    mov r10, [rbp + 200]            ; weight_ptr
    mov r11, [rbp + 216]            ; weight_grad_ptr

; @function .update_weights_loop: Loop to update weights for all layers
.update_weights_loop:
    mov rax, [rbp + 184]
    inc rax                         ; total_layers = num_hidden + 1

    cmp rsi, rax
    jge .update_biases

    mov r8, r13                     ; default: input_neurons

    test rsi, rsi
    jz .update_in_set

    mov r8, [rbp + 176]             ; hidden_neurons

; @function .update_in_set: Set input size for weight update
.update_in_set:
    mov r9, [rbp + 176]             ; hidden_neurons
    mov rax, [rbp + 184]

    cmp rsi, rax
    jl .update_weights_call

    mov r9, [rbp + 192]             ; output_neurons

; @function .update_weights_call: Apply SGD to current layer weights
.update_weights_call:
    mov rcx, r10
    mov rdx, r11
    mov rax, r8
    imul rax, r9
    mov r8, rax
    movss xmm0, [rbp + 256]         ; learning_rate
    call apply_sgd_step

    mov rax, r8
    shl rax, 2
    add r10, rax
    add r11, rax

    inc rsi
    jmp .update_weights_loop

; @function .update_biases: Update biases for all layers
.update_biases:
    xor rsi, rsi                    ; layer = 0

    mov r10, [rbp + 208]            ; bias_ptr
    mov r11, [rbp + 224]            ; bias_grad_ptr

; @function .update_biases_loop: Loop to update biases for all layers
.update_biases_loop:
    mov rax, [rbp + 184]
    inc rax

    cmp rsi, rax
    jge .epoch_done

    mov r8, [rbp + 176]             ; hidden_neurons
    mov rax, [rbp + 184]

    cmp rsi, rax
    jl .update_bias_call

    mov r8, [rbp + 192]             ; output_neurons

; @function .update_bias_call: Apply SGD to current layer biases
.update_bias_call:
    mov rcx, r10
    mov rdx, r11
    movss xmm0, [rbp + 256]
    call apply_sgd_step

    mov rax, r8
    shl rax, 2
    add r10, rax
    add r11, rax

    inc rsi
    jmp .update_biases_loop

; @function .epoch_done: Complete one epoch, continue or finish
.epoch_done:
    dec rbx
    jnz .epoch_loop

; @function .done: Label when mlp_train is done
.done:
    mov rax, [rbp + 200]
    mov rdx, [rbp + 208]

    add rsp, 144

    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rbx
    pop rbp

    ret

; =============== mlp_back_propagation ===============

; @function mlp_back_propagation: Used by mlp_train for back propagation
; @param: rcx - Input tensor pointer
; @param: rdx - Target tensor pointer
; @param: r8 - Input rows (batch size)
; @param: r9 - Input columns (input neurons)
; @param: [rbp+48] - Pointer to predictions
; @param: [rbp+56] - Amount of hidden neurons per layer
; @param: [rbp+64] - Amount of hidden layers
; @param: [rbp+72] - Output columns (output neurons)
; @param: [rbp+80] - Weight tensor pointer
; @param: [rbp+88] - Bias tensor pointer
; @param: [rbp+96] - Weight gradients
; @param: [rbp+104] - Bias gradients
; @param: [rbp+112] - Activations (flat buffer)
; @param: [rbp+120] - Deltas (flat buffer)
mlp_back_propagation:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15

    sub rsp, 120

    mov r12, r8                     ; batch_size
    mov r13, r9                     ; input_neurons
    mov [rbp - 8], rcx              ; input tensor
    mov [rbp - 16], rdx             ; target tensor
    mov [rbp - 64], r13             ; input neurons

    mov rcx, [rbp + 96]

    mov rax, r13
    imul rax, [rbp + 56]
    mov r8, rax

    mov rax, [rbp + 64]
    dec rax

    mov rbx, rax
    mov rax, [rbp + 56]

    imul rax, [rbp + 56]

    imul rax, rbx
    add r8, rax

    mov rax, [rbp + 56]
    imul rax, [rbp + 72]
    add r8, rax
    call zero_gradients

    mov rcx, [rbp + 104]
    mov rax, [rbp + 64]
    imul rax, [rbp + 56]
    add rax, [rbp + 72]
    mov r8, rax
    call zero_gradients

    mov rcx, [rbp + 48]             ; predictions
    mov rdx, [rbp - 16]             ; target tensor

    mov rax, [rbp + 64]
    imul rax, [rbp + 56]
    imul rax, r12
    mov r8, [rbp + 120]
    lea r8, [r8 + rax*4]

    mov rax, r12
    imul rax, [rbp + 72]
    mov r9, rax
    call compute_output_error

    mov rsi, [rbp + 64]             ; layer_idx = num_hidden (output layer)
    mov r13, [rbp + 80]             ; weight_ptr
    mov r14, [rbp + 88]             ; bias_ptr
    mov r15, [rbp + 96]             ; weight_grad_ptr
    call .calc_output_weight_offset

    lea r13, [r13 + rax*4]
    lea r15, [r15 + rax*4]

    mov rax, [rbp + 64]
    imul rax, [rbp + 56]
    lea r14, [r14 + rax*4]
    mov [rbp - 24], r14

; @function .bp_loop: Backpropagation loop
.bp_loop:
    test rsi, rsi
    js .done

    test rsi, rsi
    jz .bp_first_layer

    cmp rsi, [rbp + 64]
    je .bp_output_layer

    mov r8, [rbp + 56]              ; in_neurons = hidden
    mov r9, [rbp + 56]              ; out_neurons = hidden

    jmp .bp_layer_set

; @function .bp_first_layer: First layer for backpropagation
.bp_first_layer:
    mov r8, [rbp - 64]              ; in_neurons = input_neurons (original r13)
    mov r9, [rbp + 56]              ; out_neurons = hidden

    jmp .bp_layer_set

; @function .bp_output_layer: Last layer for backpropagation
.bp_output_layer:
    mov r8, [rbp + 56]              ; in_neurons = hidden
    mov r9, [rbp + 72]              ; out_neurons = output

; @function .bp_layer_set: Set up input / output neuron sizes for current layer
.bp_layer_set:
    mov [rbp - 32], r8              ; save in_neurons
    mov [rbp - 40], r9              ; save out_neurons

    cmp rsi, [rbp + 64]
    je .bp_output_acts

    test rsi, rsi
    jz .bp_use_input

    mov rax, rsi
    dec rax

    imul rax, [rbp + 56]
    imul rax, r12

    mov rcx, [rbp + 112]
    lea rcx, [rcx + rax*4]

    jmp .bp_acts_set

; @function .bp_output_acts: Get last hidden layer activation as input for output layer gradients
.bp_output_acts:
    mov rax, [rbp + 64]
    dec rax

    imul rax, [rbp + 56]
    imul rax, r12

    mov rcx, [rbp + 112]
    lea rcx, [rcx + rax*4]

    jmp .bp_acts_set

; @function .bp_use_input: Use original input tensor for layer 0 gradient comp
.bp_use_input:
    mov rcx, [rbp - 8]              ; original input

; @function .bp_acts_set: Final activation pointer for current layer gradient comp
.bp_acts_set:
    mov rax, rsi
    imul rax, [rbp + 56]

    cmp rsi, [rbp + 64]
    jl .bp_delta_hidden

    mov rax, [rbp + 64]
    imul rax, [rbp + 56]

; @function .bp_delta_hidden: Calculate delta buffer offset for current hidden layer
.bp_delta_hidden:
    imul rax, r12
    mov rdx, [rbp + 120]
    lea rdx, [rdx + rax*4]

    cmp rsi, [rbp + 64]
    je .output_layer_grads

    mov rax, rsi

    imul rax, [rbp + 56]
    imul rax, r12

    mov r10, [rbp + 112]
    lea r10, [r10 + rax*4]
    mov [rbp - 48], r10

    mov r8, rdx
    mov r9, r13
    mov rdx, r10

    mov [rsp + 40], r12             ; batch_size
    mov rax, [rbp - 32]
    mov [rsp + 48], rax             ; input_neurons
    mov [rsp + 56], rax             ; output_neurons
    mov [rsp + 64], r15             ; weight_grads
    mov rax, [rbp - 24]
    mov [rsp + 72], rax             ; bias_grads

    mov rax, rsi

    imul rax, [rbp + 56]
    imul rax, r12

    mov r11, [rbp + 120]
    lea r11, [r11 + rax*4]
    mov [rsp + 80], r11
    call mlp_backward_layer

    jmp .bp_next_layer

; @function .output_layer_grads: Handle output layer gradients
.output_layer_grads:
    mov r8, r15                     ; weight_grads
    mov r9, r12                     ; batch_size

    mov rax, [rbp - 32]
    mov [rsp + 32], rax             ; in_neurons
    mov rax, [rbp - 40]
    mov [rsp + 40], rax             ; out_neurons
    call compute_weight_gradients

    mov rcx, rdx                    ; current delta
    mov rdx, [rbp - 24]             ; bias_grads
    mov r8, r12
    mov r9, [rbp - 40]              ; out_neurons
    call compute_bias_gradients

; @function .bp_next_layer: Update weight / bias pointers and move to previous layer
.bp_next_layer:
    mov rax, [rbp - 32]
    imul rax, [rbp - 40]
    shl rax, 2
    sub r13, rax
    sub r15, rax

    mov rax, [rbp - 32]
    shl rax, 2
    sub r14, rax
    sub [rbp - 24], rax

    dec rsi
    jmp .bp_loop

; @function .done: Label when mlp_back_propagation is done
.done:
    add rsp, 120

    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp

    ret

; @function .calc_output_weight_offset: Calculate weight offset for output layer
.calc_output_weight_offset:
    push rcx
    push rbx

    mov rax, r13
    imul rax, [rbp + 56]

    mov rcx, 1

; @function .cowo_loop: Accumulate weight offsets for all hidden layers
.cowo_loop:
    cmp rcx, [rbp + 64]
    jge .cowo_done

    mov rbx, [rbp + 56]
    imul rbx, [rbp + 56]
    add rax, rbx

    inc rcx
    jmp .cowo_loop

; @function .cowo_done: Label when cowo_loop is done
.cowo_done:
    pop rbx
    pop rcx

    ret

; =============== mlp_forward_layer ===============

; @function mlp_forward_layer: Forward pass for one hidden layer (No softmax)
; @param: rcx - Input tensor pointer
; @param: rdx - Weight tensor pointer
; @param: r8 - Bias tensor pointer
; @param: r9 - Output tensor pointer (pre-allocated)
; @param: [rbp+40] - Batch size
; @param: [rbp+48] - Input neurons
; @param: [rbp+56] - Output neurons
; @param: [rbp+64] - Apply dropout (0 or 1)
; @param: [rbp+72] - Dropout rate
; @return: rax - Pointer to output tensor
mlp_forward_layer:
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

    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9

    mov rbx, [rbp + 72]     ; batch size
    mov r10, [rbp + 80]     ; input_neurons
    mov r11, [rbp + 88]     ; output_neurons

    xor rsi, rsi            ; i = 0

; @function .row_loop: Apply activation to each row
.row_loop:
    cmp rsi, rbx
    jge .apply_relu

    mov rax, rsi
    imul rax, r11
    lea rdi, [r15, rax*4]

    xor r8, r8

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
    xor rcx, rcx

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

    xor r8, r8

; @function .j_loop_simd: Loop through current row and SIMD scalar mult (4 SIMD)
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
.apply_relu:
    mov rcx, r15
    mov rax, rbx
    imul rax, r11
    mov r8, rax

    call relu

    cmp qword [rbp + 96], 0
    je .done

    mov rcx, r15
    mov rax, rbx
    imul rax, r11
    mov r8, rax

    movss xmm0, [rbp + 104]

    call apply_dropout

; @function .done: Label when mlp_forward_layer is done
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

; =============== mlp_backward_layer ===============

; @function mlp_backward_layer: Backward pass for one hidden layer
; @param: rcx - Input activations (from previous layer)
; @param: rdx - Current layer activations (for relu_derivative)
; @param: r8 - Delta from next layer
; @param: r9 - Weights of next layer
; @param: [rbp+40] - Batch size
; @param: [rbp+48] - Input neurons (this layer)
; @param: [rbp+56] - Output neurons (next layer)
; @param: [rbp+64] - Weight gradient buffer (this layer)
; @param: [rbp+72] - Bias gradient buffer (this layer)
; @param: [rbp+80] - Delta buffer (this layer, OUTPUT)
; @return: rax - Pointer to this layer's delta
mlp_backward_layer:
    push rbp
    mov rbp, rsp

    push r12
    push r13

    sub rsp, 48

    mov r12, rcx               ; input activations
    mov r13, rdx               ; current activations

    mov rcx, r8                ; next delta
    mov rdx, r9                ; next weights
    mov r8, [rbp + 88]         ; this delta (output)
    mov r9, [rbp + 48]         ; batch size

    mov rax, [rbp + 56]
    mov [rsp + 32], rax        ; input neurons
    mov rax, [rbp + 64]
    mov [rsp + 40], rax        ; output neurons
    call compute_hidden_error

    mov rcx, r13               ; activations
    mov rdx, [rbp + 88]        ; this delta
    mov rax, [rbp + 48]
    imul rax, [rbp + 56]
    mov r8, rax                ; batch * neurons
    call relu_derivative

    mov rcx, r12               ; input activations
    mov rdx, [rbp + 88]        ; this delta
    mov r8, [rbp + 72]         ; weight gradients
    mov r9, [rbp + 48]         ; batch size

    mov rax, [rbp + 56]
    mov [rsp + 32], rax        ; input neurons
    mov [rsp + 40], rax        ; output neurons
    call compute_weight_gradients

    mov rcx, [rbp + 88]        ; this delta
    mov rdx, [rbp + 80]        ; bias gradients
    mov r8, [rbp + 48]         ; batch size
    mov r9, [rbp + 56]         ; input neurons
    call compute_bias_gradients

    mov rax, [rbp + 88]

    add rsp, 48

    pop r13
    pop r12
    pop rbp

    ret

; =============== compute_weight_gradients ===============

; @function compute_weight_gradients: Compute weight gradients (in-place)
; @param rcx - Pointer to activations (Input to this layer)
; @param rdx - Pointer to errors (Delta from next layer)
; @param r8  - Pointer to the specific block in grad_base_ptr
; @param r9  - Batch Size
; @param [rbp+40] - Input neurons (columns in X)
; @param [rbp+48] - Output neurons (columns in Delta)
compute_weight_gradients:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi
    push rdi
    push r12
    push r13

    sub rsp, 40

    mov r10, [rbp + 56]
    mov r11, [rbp + 64]

    xor rsi, rsi           ; b = 0 (batch loop)

; @function .batch_loop: Check if the current batch loop is done
.batch_loop:
    cmp rsi, r9
    jge .done

    xor rbx, rbx           ; i = 0 (input neuron loop)

; @function .input_loop: Compute the current batch
.input_loop:
    cmp rbx, r10
    jge .next_batch

    mov rax, rsi
    imul rax, r10
    add rax, rbx

    movss xmm0, [rcx + rax*4]
    shufps xmm0, xmm0, 0
    
    mov rax, rbx
    imul rax, r11
    lea r12, [r8 + rax*4]

    mov rax, rsi
    imul rax, r11
    lea r13, [rdx + rax*4]

    xor rdi, rdi           ; j = 0

; @function .output_loop_simd: SIMD Matmul (4 SIMD)
.output_loop_simd:
    mov rax, r11
    sub rax, rdi

    cmp rax, 4
    jl .output_loop_scalar

    movups xmm1, [r13 + rdi*4]
    mulps xmm1, xmm0

    movups xmm2, [r12 + rdi*4]
    addps xmm2, xmm1
    movups [r12 + rdi*4], xmm2

    add rdi, 4
    jmp .output_loop_simd

; @function .output_loop_scalar: Cleanup scalar and calculate next output
.output_loop_scalar:
    cmp rdi, r11
    jge .next_input

    movss xmm1, [r13 + rdi*4]
    mulss xmm1, xmm0
    addss xmm1, [r12 + rdi*4]
    movss [r12 + rdi*4], xmm1

    inc rdi
    jmp .output_loop_scalar

; @function .next_input: Jump to next input
.next_input:
    inc rbx
    jmp .input_loop

; @function .next_batch: Jump to next batch
.next_batch:
    inc rsi
    jmp .batch_loop

; @function .done: Label when mlp_back_propagation is done
.done:
    add rsp, 40

    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp

    ret

; =============== compute_bias_gradients ===============

; @function compute_bias_gradients: Compute bias gradients by summing deltas across batch
; @param rcx - Pointer to delta buffer
; @param rdx - Pointer to bias gradient buffer
; @param r8 - Batch size
; @param r9 - Output neurons
compute_bias_gradients:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi

    xor rbx, rbx           ; j = 0 (output neuron loop)

; @function .neuron_loop: Loop through current neuron layer
.neuron_loop:
    cmp rbx, r9
    jge .done

    xorps xmm0, xmm0
    xor rsi, rsi           ; b = 0 (batch loop)

; @function .batch_loop: Loop through current batch
.batch_loop:
    cmp rsi, r8
    jge .store_gradient

    mov rax, rsi
    imul rax, r9
    add rax, rbx
    
    movss xmm1, [rcx + rax*4]
    addss xmm0, xmm1

    inc rsi
    jmp .batch_loop

; @function .store_gradient: Store current gradient
.store_gradient:
    movss xmm1, [rdx + rbx*4]
    addss xmm1, xmm0
    movss [rdx + rbx*4], xmm1

    inc rbx
    jmp .neuron_loop

; @function .done: Label when compute_bias_gradients is done
.done:
    pop rsi
    pop rbx
    pop rbp

    ret

; =============== compute_hidden_error ===============

; @function compute_hidden_error: Computer error in hidden layers (in-place)
; @param rcx - Pointer to current Delta
; @param rdx - Pointer to Weights
; @param r8 - Pointer to Previous Delta Buffer
; @param r9 - Batch Size
; @param [rbp+40] - Input Neurons
; @param [rbp+48] - Output Neurons
compute_hidden_error:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi
    push rdi
    push r12
    push r13

    sub rsp, 24

    mov r10, [rbp + 56]
    mov r11, [rbp + 64]

    xor rsi, rsi           ; b = 0

; @function .batch_loop: Loop to start current batch loop
.start_batch_loop:
    cmp rsi, r9
    jge .done

    xor rbx, rbx           ; i = 0 (input neurons)

; @function .next_batch: Start the next batch
.next_batch:
    inc rsi
    jmp .start_batch_loop

; @function .in_loop: Loop to calculate current batch hidden error
.in_loop:
    cmp rbx, r10
    jge .next_batch
    
    mov rax, rsi
    imul rax, r11
    lea r12, [rcx + rax*4]

    mov rax, rbx
    imul rax, r11
    lea r13, [rdx + rax*4]

    xorps xmm0, xmm0

    xor rdi, rdi           ; j = 0 (output neurons)

; @function .out_loop_simd: SIMD Dot Product Calculation (4 SIMD)
.out_loop_simd:
    mov rax, r11
    sub rax, rdi

    cmp rax, 4
    jl .out_loop_scalar

    movups xmm1, [r12 + rdi*4]
    movups xmm2, [r13 + rdi*4]
    mulps xmm1, xmm2
    addps xmm0, xmm1

    add rdi, 4
    jmp .out_loop_simd

; @function .out_loop_scalar: Calculate output loop scalar
.out_loop_scalar:
    cmp rdi, r11
    jge .horizontal_add

    movss xmm1, [r12 + rdi*4]
    mulss xmm1, [r13 + rdi*4]
    addss xmm0, xmm1

    inc rdi
    jmp .out_loop_scalar

; @function .horizontal_add: SIMD Horizontal add
.horizontal_add:
    haddps xmm0, xmm0
    haddps xmm0, xmm0

    mov rax, rsi
    imul rax, r10
    add rax, rbx
    movss [r8 + rax*4], xmm0

    inc rbx
    jmp .in_loop

; @function .done: Label when compute_hidden_error is done
.done:
    add rsp, 24

    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    pop rbp

    ret

; =============== zero_gradients ===============

; @function zero_gradients: Zero gradients (in-place)
; @param rcx - Pointer to gradient buffer (grad_base_ptr)
; @param r8  - Total number of floats to zero
zero_gradients:
    xorps xmm0, xmm0
    xor r10, r10

    mov rax, r8
    and rax, -4

; @function .simd_loop: SIMD Loop to clear all gradients
.simd_loop:
    cmp r10, rax
    jge .scalar_tail

    movups [rcx + r10*4], xmm0

    add r10, 4
    jmp .simd_loop

; @function .scalar_tail: Clean up any remaining gradients
.scalar_tail:
    cmp r10, r8
    jge .done

    mov dword [rcx + r10*4], 0

    inc r10
    jmp .scalar_tail

; @function .done: Label when zero_gradients is done
.done:
    ret

; =============== compute_output_error ===============

; @function compute_output_error: SIMD "Prediction - Target" (4 SIMD) (in-place)
; @param rcx - Pointer to Final layer activations (Predictions)
; @param rdx - Pointer to Target tensor (Labels)
; @param r8  - Pointer to Destination (First Delta Buffer)
; @param r9  - Total number of floats (Batch * Output_Neurons)
compute_output_error:
    xor r10, r10

    mov rax, r9
    and rax, -4

; @function .simd_loop: 4 SIMD Scalar Subtraction
.simd_loop:
    cmp r10, rax
    jge .scalar_tail

    movups xmm0, [rcx + r10*4]
    movups xmm1, [rdx + r10*4]

    subps xmm0, xmm1
    movups [r8 + r10*4], xmm0

    add r10, 4
    jmp .simd_loop

; @function .scalar_tail: Subtract remaining scalars
.scalar_tail:
    cmp r10, r9
    jge .done

    movss xmm0, [rcx + r10*4]
    subss xmm0, [rdx + r10*4]

    movss [r8 + r10*4], xmm0

    inc r10
    jmp .scalar_tail

; @function .done: Label when compute_output_error is done
.done:
    ret

; =============== apply_sgd_step ===============

; @function apply_sgd_step: Apply optimizer (in-place)
; @param rcx - Pointer to Weight tensor
; @param rdx - Pointer to Gradient tensor
; @param r8  - Total number of elements
; @param xmm0 - Learning Rate (scalar float)
apply_sgd_step:
    shufps xmm0, xmm0, 0
    xor r10, r10

    mov rax, r8
    and rax, -4

; @function .simd_loop: 4 SIMD Scalar Compute
.simd_loop:
    cmp r10, rax
    jge .scalar_tail

    movups xmm1, [rdx + r10*4]
    mulps xmm1, xmm0

    movups xmm2, [rcx + r10*4]
    subps xmm2, xmm1
    movups [rcx + r10*4], xmm2

    add r10, 4
    jmp .simd_loop

; @function .scalar_tail: Compute remaining scalars
.scalar_tail:
    cmp r10, r8
    jge .done

    movss xmm1, [rdx + r10*4]
    mulss xmm1, xmm0
    movss xmm2, [rcx + r10*4]

    subss xmm2, xmm1
    movss [rcx + r10*4], xmm2

    inc r10
    jmp .scalar_tail

; @function .done: Label when apply_sgd_step is done
.done:
    ret
