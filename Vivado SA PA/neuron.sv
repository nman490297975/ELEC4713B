`timescale 1ns / 1ps

module neuron #(BITWIDTH = 16, BW = BITWIDTH+2-1)(

  input                         clk,				 
  input 				                rst,				 
  input                         clear,

  input                         en,         //Enbale forward propogation

  input [BW:0]                  weight,		  
  input [BW:0]                  data,			  

  output logic[BW:0]	          accum 

);
  logic [BW:0] prod;
  logic [BW:0] accum_r;
    
  FPMult_WRAPPER FPMult_neuron(.X(weight), .Y(data), .R(prod));
  FPADD_WRAPPER FPAdd_neuron(.X(prod), .Y(accum), .R(accum_r)); 

  //Sequential Logic for Accumulation
  always_ff @(posedge clk) begin
    
    if (rst) begin      
        accum <= 18'b00000000000000000;
    end
    //Clear Accumulation
    else if (clear) begin
      accum <= 18'b00000000000000000;
    end
    else if (en) begin
      accum <= accum_r; 
    end
    else ;
  end
  
endmodule
