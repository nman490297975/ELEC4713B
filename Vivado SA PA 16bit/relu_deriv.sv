`timescale 1ns / 1ps

`define FLOAT_1 18'b000011110000000000
// `define FLOAT_1 18'b111111111111111111 // used in debugging synthesis/implementation

module relu_deriv #(BITWIDTH = 16, BW = BITWIDTH+2-1)(
  
    input                           clk,
    input                           rst,
    input                           enable,

    input                           in, 
    output logic [BW:0]             out

); 

    logic [BW:0]                    temp_out; 

    always_comb begin
        case (in)
            1: temp_out = 0;
            0: temp_out = `FLOAT_1;
        endcase
    end 
 
    always_ff @(posedge clk or posedge rst) begin

        if(rst) begin
            out <= 0;
        end else if (enable) begin
            out <= temp_out;
        end
    
    end

endmodule 
