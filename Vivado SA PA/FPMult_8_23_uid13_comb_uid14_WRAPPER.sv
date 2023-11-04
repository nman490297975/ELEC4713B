`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07.10.2023 15:40:40
// Design Name: 
// Module Name: FPMult_8_23_uid13_comb_uid14_WRAPPER
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


module FPMult_8_23_uid13_comb_uid14_WRAPPER(

    input                [31:0] X,
    input                [31:0] Y,
    input                [31:0] C,
    input                negateAB,
    input                negateC,
    input          [1:0] RndMode,
    output logic         [31:0] R


);

  // Instantiate the top VHDL module
  IEEEFPFMA_8_23_comb_uid2 FPMult_inst (
    .A(X),
    .B(Y),
    .C(C), 
    .negateAB(negateAB),
    .negateC(negateC),
    .RndMode(RndMode),
    .R(R)
  );

endmodule
