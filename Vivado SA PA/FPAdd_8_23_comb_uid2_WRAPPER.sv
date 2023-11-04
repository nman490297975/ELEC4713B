`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07.10.2023 15:38:32
// Design Name: 
// Module Name: FPAdd_8_23_comb_uid2_WRAPPER
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


module FPAdd_8_23_comb_uid2_WRAPPER(

    input                [31:0] X,
    input                [31:0] Y,
    output               [31:0] R

);

  // Instantiate the top VHDL module
  IEEEFPAdd_8_23_comb_uid2 FPAdd_inst (
    .X(X),
    .Y(Y),
    .R(R)
  );

endmodule
