`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.10.2023 22:59:04
// Design Name: 
// Module Name: FPADD_11bit_WRAPPER
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


module FPADD_WRAPPER(

    input               [17:0] X,
    input               [17:0] Y,

    output logic        [17:0] R
);

FPADD_11bit FPADD_inst1(.X(X), .Y(Y), .R(R));

endmodule
