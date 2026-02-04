extern mlp_feed_forward
extern mlp_train
extern mlp_back_propagation

extern printf
extern exit

section .text
    global main

main:
    sub rsp, 40
    add rsp, 40

    xor ecx, ecx

    call exit
