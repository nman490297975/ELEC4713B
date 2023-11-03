`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.10.2023 22:59:04
// Design Name: 
// Module Name: FPMult_11bit_WRAPPER
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module FPMult_WRAPPER(

    input               [17:0] X,
    input               [17:0] Y,

    output logic        [17:0] R
);

FPMult_11bit FPMult_inst1(.X(X), .Y(Y), .R(R));

endmodule
