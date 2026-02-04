default rel

; @extern: MLP Settings

extern input_neurons
extern hidden_neurons
extern hidden_layers
extern output_neurons
extern epochs
extern learning_rate
extern dropout_rate
extern batch_size

; @extern: Math Functions

extern relu
extern softmax
extern cross_entropy_loss

; @section: Global labels
section .text
    global mlp_feed_forward
    global mlp_train
    global mlp_back_propagation

; =============== PUBLIC LABELS ===============

; @function: mlp_feed_forward - Feed forward a matrix of input values
; @param: rcx - Pointer to the matrix
; @param: rax - Pointer to the weight matrix
; @param: rdx - Pointer to the bias vector
; @param: [rsp+40] - Source matrix rows
; @param: [rsp+48] - Source matrix columns
; @param: al - 1 to enable dropout, 0 to disable
; @return: rax - Pointer to the output matrix
mlp_feed_forward:

; @function: mlp_train - Train the MLP using Inputs and Targets
; @param: rcx - Source to the input matrix
; @param: r9 - Source to the target matrix
; @return: rax - Pointer to the weight matrix
; @return: rax[rsp+8] - Pointer to the bias vecto
mlp_train:

; @function: mlp_back_propagation - Used by mlp_train for back propagation
; @param: rcx - Source to the input matrix
; @param: r9 - Source to the target matrix
; @return: rax - Pointer to the updated weight matrix
; @return: rax[rsp+8] - Pointer to the updated bias vector
mlp_back_propagation:
