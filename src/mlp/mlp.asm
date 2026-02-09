default rel

; @extern: Math Functions

extern relu
extern relu_derivative
extern softmax
extern cross_entropy_loss

; @extern: External Libraries

extern apply_dropout

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
; @param: [rbp+56] - Output columns (output neurons)
; @param: [rbp+64] - Apply dropout (0 or 1)
; @param: [rbp+72] - If dropout, dropout rate
; @return: rax - Pointer to output tensor
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

    mov rbx, [rbp + 56]     ; input_rows (batch size)
    mov r10, [rbp + 64]     ; input_cols
    mov r11, [rbp + 72]     ; output_cols

    xor rsi, rsi            ; i = 0

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

; @function .done: Label when mlp_feed_forward is done
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
; @param: rcx - Source to the input tensor
; @param: rdx - Source to the target tensor
; @param: r8 - Input rows (batch size)
; @param: r9 - Input columns (input neurons)
; @param: [rbp+40] - Output columns (output neurons)
; @param: [rbp+48] - Pointer to the weight tensor
; @param: [rbp+56] - Pointer to the bias tensor
; @param: [rbp+64] - Pointer to the specific gradient block (grad_base_ptr)
; @param: [rbp+72] - Pointer to the previous delta buffer
; @param: [rbp+80] - Amount of Epochs
; @param: [rbp+88] - Learning Rate
; @param: [rbp+96] - Apply dropout (0 or 1)
; @param: [rbp+104] - If dropout, dropout rate
; @return: rax - Pointer to the weight tensor
; @return: rdx - Pointer to the bias tensor
mlp_train:
    push rbp
    mov rbp, rsp

    push rbx
    push rsi
    push r12
    push r13
    push r14
    push r15

    sub rsp, 112

    mov r12, r8                 ; batch size
    mov r13, r9                 ; input neurons
    mov r14, rcx                ; input tensor
    mov r15, rdx                ; target tensor

    mov rbx, [rbp + 96]         ; epochs

    test rbx, rbx
    jz .done

; @function .epoch_loop: Epoch training loop
.epoch_loop:
    mov rcx, r14                 ; input tensor
    mov rdx, [rbp + 64]          ; weight tensor
    mov r8, [rbp + 72]           ; bias tensor
    mov r9, [rbp + 88]           ; delta buffer

    mov rax, r12
    mov [rsp + 40], rax          ; batch size
    mov rax, r13
    mov [rsp + 48], rax          ; input neurons
    mov rax, [rbp + 56]          ; output neurons
    mov [rsp + 56], rax
    mov rax, [rbp + 112]         ; apply dropout
    mov [rsp + 64], rax
    movss xmm0, [rbp + 120]      ; dropout rate
    movss [rsp + 72], xmm0 
    call mlp_feed_forward

    mov [rsp + 48], rax          ; save predictions

    mov rcx, r14                 ; input tensor
    mov rdx, r15                 ; target
    mov r8, r12                  ; batch size
    mov r9, r13                  ; input neurons

    mov rax, [rsp + 48]          ; predictions
    mov [rsp + 32], rax
    mov rax, [rbp + 56]
    mov [rsp + 40], rax          ; output neurons
    mov rax, [rbp + 64]
    mov [rsp + 48], rax          ; weight_base_ptr
    mov rax, [rbp + 72]
    mov [rsp + 56], rax          ; bias_base_ptr
    mov rax, [rbp + 80]
    mov [rsp + 64], rax          ; grad_base_ptr
    mov rax, [rbp + 64]
    mov [rsp + 72], rax          ; weight tensor for layer
    mov rax, [rbp + 88]
    mov [rsp + 80], rax          ; delta buffer
    call mlp_back_propagation

    mov rcx, [rbp + 64]          ; weight tensor
    mov rdx, [rbp + 80]          ; grad_base_ptr

    mov rax, r13                 ; input_neurons
    mov r11, [rbp + 56]          ; output_neurons
    imul rax, r11                
    mov r8, rax
    movss xmm0, [rbp + 104]      ; learning rate
    call apply_sgd_step

    mov rcx, [rbp + 72]          ; bias tensor
    mov rdx, [rbp + 80]          ; grad_base_ptr

    mov rax, r13
    imul rax, r11
    lea rdx, [rdx + rax*4]       ; bias gradients offset
    mov r8, r11                  ; output_neurons
    movss xmm0, [rbp + 104]      ; learning rate
    call apply_sgd_step

    dec rbx
    jnz .epoch_loop

; @function .done: Label when mlp_train is done
.done:
    mov rax, [rbp + 64]
    mov rdx, [rbp + 72]

    add rsp, 112

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
; @param: rcx - Source to the input tensor
; @param: rdx - Source to the target tensor
; @param: r8 - Batch size
; @param: r9 - Input neurons
; @param: [rbp+48] - Pointer to predictions
; @param: [rbp+56] - Output neurons
; @param: [rbp+64] - Pointer to the specific weight tensor (weight_base_ptr)
; @param: [rbp+72] - Pointer to the specific bias tensor (bias_base_ptr)
; @param: [rbp+80] - Pointer to the specific gradient block (grad_base_ptr)
; @param: [rbp+88] - Pointer to the weight tensor for this layer
; @param: [rbp+96] - Pointer to the previous delta buffer
; @return: rax - Pointer to the updated weight tensor
; @return: rdx - Pointer to the updated bias tensor
mlp_back_propagation:
    push rbp
    mov rbp, rsp

    push rbx
    push r12
    push r13
    push r14
    push r15

    sub rsp, 72

    mov r12, r8                  ; batch size
    mov r13, r9                  ; input neurons
    mov r14, [rbp + 56]          ; output neurons
    mov r15, [rbp + 48]          ; predictions

    mov rcx, [rbp + 80]          ; grad_base_ptr
    mov rax, r13
    imul rax, r14
    mov r8, rax
    call zero_gradients

    mov rcx, [rbp + 80]          ; grad_base_ptr
    mov rax, r13
    imul rax, r14
    lea rcx, [rcx + rax*4]       ; bias gradients offset
    mov r8, r14                  ; output neurons
    call zero_gradients

    mov rcx, r15                 ; predictions
    mov rdx, rdx                 ; tensor target
    mov r8, [rbp + 96]           ; delta buffer
    mov rax, r12
    imul rax, r14                ; batch * output_neurons
    mov r9, rax
    call compute_output_error

    mov rcx, rcx                 ; tensor input
    mov rdx, [rbp + 96]          ; delta buffer
    mov r8, [rbp + 80]           ; grad_base_ptr 
    mov r9, r12                  ; batch size
    mov [rsp + 40], r13          ; input neurons
    mov [rsp + 48], r14          ; output neurons
    call compute_weight_gradients

    mov rcx, [rbp + 96]          ; delta buffer
    mov rdx, [rbp + 80]          ; grad_base_ptr
    mov rax, r13
    imul rax, r14
    lea rdx, [rdx + rax*4]       ; bias gradients offset
    mov r8, r12                  ; batch size
    mov r9, r14                  ; output neurons
    call compute_bias_gradients

    mov rax, [rbp + 88]
    mov rdx, [rbp + 72]

    add rsp, 72

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp

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

    xor rsi, rsi

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

    sub rsp, 40

    mov r12, rcx               ; input activations
    mov r13, rdx               ; current activations

    mov rcx, r8                ; next delta
    mov rdx, r9                ; next weights
    mov r8,  [rbp + 96]        ; this delta (output)
    mov r9,  [rbp + 56]        ; batch size

    mov rax, [rbp + 64]        ; neurons
    mov [rsp + 48], rax        ; input neurons
    mov [rsp + 56], rax        ; output neurons

    call compute_hidden_error

    mov rcx, r13               ; activations
    mov rdx, [rbp + 96]        ; this delta
    mov rax, [rbp + 56]
    imul rax, [rbp + 64]
    mov r8, rax                ; batch * neurons

    call relu_derivative

    mov rcx, r12               ; input activations
    mov rdx, [rbp + 96]        ; this delta
    mov r8, [rbp + 80]         ; weight gradients
    mov r9, [rbp + 56]         ; batch size

    mov rax, [rbp + 64]
    mov [rsp + 48], rax        ; input neurons
    mov [rsp + 56], rax        ; output neurons

    call compute_weight_gradients

    mov rcx, [rbp + 96]        ; this delta
    mov rdx, [rbp + 88]        ; bias gradients
    mov r8, [rbp + 56]         ; batch size
    mov r9, [rbp + 64]         ; neurons

    call compute_bias_gradients

    mov rax, [rbp + 96]        ; return delta

    add rsp, 40

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
