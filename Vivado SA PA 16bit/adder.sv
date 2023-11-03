    `timescale 1ns / 1ps

module adder #(BITWIDTH = 16, BW = BITWIDTH+2-1) (

    input                           clk,
    input                           rst,
    input                           enable,
    input                           clear,

    input [BW:0]                    acc,
    output logic [BW:0]             sum

);

    logic [BW:0] prod;
    logic [BW:0] mult;

    FPADD_WRAPPER FPAdd_adder(.X(acc), .Y(prod), .R(mult));

    always_comb begin
        prod = sum;
    end

    always_ff @(posedge clk) begin

    if (rst) begin
        sum <= 18'b0;
    end

    else if (clear) begin
        sum <= 18'b0;
    end

    else if (enable) begin
        sum <= mult; 
    end
  end

endmodule
