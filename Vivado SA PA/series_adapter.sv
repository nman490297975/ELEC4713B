`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
//
// Engineer: Nicholas Manning and Niyana Yohannes
// 
// Design Name: CNN_SA
// Module Name: series_adapter
// Project Name: Convolutional Adapter
// Target Devices: ZCU111
// Tool Versions: Vivado Design Suite 2022 and 2023
// Description: Proof of concept for a hardware convolutional layer serial adapter using a small CNN (SSCNN by Jiong Si)
// 
// Revision 1.00 - File Published
//
//////////////////////////////////////////////////////////////////////////////////

/*------------------------------------------------------------------------------
--  USYD Code Citation
--
--	The forward propgagtion logic was built off code by Jiong Si presented in Appendix A of https://digitalscholarship.unlv.edu/cgi/viewcontent.cgi?article=4849&context=thesesdissertations
--	The backpropgation code was built off code for an MLP by Team-SDK-545 hosted by member Kais Kudrolli https://github.com/kkudrolli/Team-SDK-545
--  
--  Niyana Yohannes was primarily responsible for the initial RTL and for the initial backpropgation logic.  Also primarily responsible for the SA adapter.
--  Nicholas Manning was primarily responsible for the final design of the RTL adapter logic, introduction of cross-entropy, administration of the document, changes made after debugging to obtain a synthesizable and implementable design and for the PA adapter.
--
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
--  
--  Structure
--
--  1. Define Constants
--  
--  External (I/O)
--  2. Control Signals
--  3. Data Signals
--  
--  Internal
--  4. Control Signals / Constants
--
--  FSM
--  5. FSM Logic
--  6. FSM Indexer
--
--  Hardware Generation
--  7. Forward Propagation Generation
--  8. Backward Propagation Generation
--  
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
--  1. Define Constants
------------------------------------------------------------------------------*/

`define BIT_WIDTH 16                   

module cnn_topmodule #(FMAP_SIZE_SA = 24*24, FMAP_WIDTH_SA = 24, FMAP_CHANNELS = 6, SA_SIZE = 24*24, SA_KERNELS = 1, BW = `BIT_WIDTH+2-1) (
  
    /*------------------------------------------------------------------------------
    --  2. I/O Control Signals
    ------------------------------------------------------------------------------*/

    input logic                                         clk, rst, do_fp, 
    output logic                                        done_FP, done_BP,

    /*------------------------------------------------------------------------------
    --  3. I/O Data Signals
    ------------------------------------------------------------------------------*/

    input logic [SA_KERNELS*FMAP_SIZE_SA-1:0]  		[BW:0]    fp_fmap,              

    input logic [SA_KERNELS*FMAP_CHANNELS-1:0]    [BW:0]    weights_SA,   
    input logic [SA_KERNELS-1:0]                  [BW:0]    biases_SA,    

    input logic [SA_KERNELS*FMAP_SIZE_SA-1:0]      [BW:0]    error_IN,     

    output logic [SA_KERNELS*SA_SIZE-1:0]         [BW:0]    adapter_z,    

    output logic [SA_KERNELS*SA_SIZE-1:0]         [BW:0]    acc_deriv_SA, 
    output logic [SA_KERNELS*SA_SIZE-1:0]         [BW:0]    error_OUT_SA, 
    output logic [SA_KERNELS*FMAP_CHANNELS-1:0]   [BW:0]    bpWchange_SA, 
    output logic [SA_KERNELS-1:0]                 [BW:0]    bpBchange_SA 

);

    /*------------------------------------------------------------------------------
    --  4. Control Signals and Constants
    ------------------------------------------------------------------------------*/

    logic                                   clear;

    logic [SA_SIZE-1:0] [BW:0]              adapter_z0; 
	logic [SA_SIZE-1:0] [BW:0]              convadapter_z; 
	logic [SA_SIZE-1:0] [BW:0]              convadapter_actv;

	logic [SA_SIZE-1:0] [BW:0]              actv_in_x_error_out0, actv_in_x_error_out1, actv_in_x_error_out2, actv_in_x_error_out3, actv_in_x_error_out4, actv_in_x_error_out5;

    logic                                   enable_conv0, enable_convadapt_s, enable_convadapt_d, enable_accum_bp_db;
	logic 									enable_accum_bp_dw0, enable_accum_bp_dw1, enable_accum_bp_dw2, enable_accum_bp_dw3, enable_accum_bp_dw4, enable_accum_bp_dw5;
	integer                                 k, j; 

	enum logic [3:0] {S_FPIDLE, S_FPROP_CONV0, S_FPROP_CONV0_b, S_FPROP_CONV0_s, S_FPROP_CONVADAPT_s, S_BPROP_CONVADAPT_d, S_BPROP_ACCUM_db, 
						S_BPROP_ACCUM_dw0, S_BPROP_ACCUM_dw1, S_BPROP_ACCUM_dw2, S_BPROP_ACCUM_dw3, S_BPROP_ACCUM_dw4, S_BPROP_ACCUM_dw5, S_FPDONE} fpcs, fpns;

    /*------------------------------------------------------------------------------
    --  10. FSM State Logic
    ------------------------------------------------------------------------------*/

	always_ff @(posedge clk, posedge rst)
		if (rst) fpcs <= S_FPIDLE;
		else fpcs <= fpns;
	

	always_comb begin
		done_FP = 1'b0;
		done_BP = 1'b0;
		clear = 1'b0;
		enable_conv0 = 1'b0;
		enable_convadapt_s = 1'b0;
		enable_convadapt_d = 1'b0;
		enable_accum_bp_db = 1'b0;
		enable_accum_bp_dw0 = 1'b0;
		enable_accum_bp_dw1 = 1'b0;
		enable_accum_bp_dw2 = 1'b0;
		enable_accum_bp_dw3 = 1'b0;
		enable_accum_bp_dw4 = 1'b0;
		enable_accum_bp_dw5 = 1'b0;

		case (fpcs)

			S_FPIDLE: begin
				clear = do_fp;
				fpns = (do_fp) ? S_FPROP_CONV0 : S_FPIDLE;
			end

			S_FPROP_CONV0: begin
				enable_conv0 = 1'b1;
				fpns = (k == FMAP_CHANNELS - 1) ? S_FPROP_CONV0_b : S_FPROP_CONV0;
			end

			S_FPROP_CONV0_b: begin
				fpns = S_FPROP_CONV0_s;
			end
    
			S_FPROP_CONV0_s: begin
				fpns = S_FPROP_CONVADAPT_s;
			end

            S_FPROP_CONVADAPT_s: begin
				done_FP = 1'b1;
				enable_convadapt_s = 1'b1;
				fpns = S_BPROP_CONVADAPT_d;
			end

			S_BPROP_CONVADAPT_d: begin
				enable_convadapt_d = 1'b1;
				fpns = S_BPROP_ACCUM_db;
			end

			S_BPROP_ACCUM_db: begin
				enable_accum_bp_db = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw0 : S_BPROP_ACCUM_db;
			end

			S_BPROP_ACCUM_dw0: begin
				enable_accum_bp_dw0 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw1 : S_BPROP_ACCUM_dw0;
			end
			S_BPROP_ACCUM_dw1: begin
				enable_accum_bp_dw1 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw2 : S_BPROP_ACCUM_dw1;
			end
			S_BPROP_ACCUM_dw2: begin
				enable_accum_bp_dw2 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw3 : S_BPROP_ACCUM_dw2;
			end
			S_BPROP_ACCUM_dw3: begin
				enable_accum_bp_dw3 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw4 : S_BPROP_ACCUM_dw3;
			end
			S_BPROP_ACCUM_dw4: begin
				enable_accum_bp_dw4 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw5 : S_BPROP_ACCUM_dw4;
			end
			S_BPROP_ACCUM_dw5: begin
				enable_accum_bp_dw5 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_FPDONE : S_BPROP_ACCUM_dw5;
			end
			S_FPDONE: begin
				done_BP = 1'b1;
				fpns = S_FPIDLE;
			end

			default: fpns = S_FPIDLE;

			endcase
	end

    /*------------------------------------------------------------------------------
    --  11. FSM Indexer
    ------------------------------------------------------------------------------*/

	always_ff @(posedge clk, posedge rst) begin

		if (rst) begin
			k <= '0;
			j <= '0;
		end
		
		else begin
			case (fpcs)

				S_FPIDLE: begin
					k <= '0;
					j <= '0;
				end

				S_FPROP_CONV0: begin
					k <= (k < FMAP_CHANNELS - 1) ? (k + 1) : 0;
				end

				S_BPROP_ACCUM_db: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end

				S_BPROP_ACCUM_dw0: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				S_BPROP_ACCUM_dw1: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				S_BPROP_ACCUM_dw2: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				
				S_BPROP_ACCUM_dw3: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				S_BPROP_ACCUM_dw4: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				S_BPROP_ACCUM_dw5: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end

				default: ;

			endcase
		end
	end

	genvar m;
	generate
		for (m = 0; m < SA_SIZE; m++) begin: ADAPTER

            neuron neuron_conv (.clk(clk), .rst(rst), .clear(clear), .en(enable_conv0), .weight(weights_SA[k]), .data(fp_fmap[m]), .accum(adapter_z0[m]));
            FPADD_WRAPPER FPADD_bias(.X(adapter_z0[m]), .Y(biases_SA[0]), .R(adapter_z[m])); 

		end
	endgenerate

	generate
		for (m = 0; m < SA_SIZE; m++) begin: CONV_ADAPTER

            FPADD_WRAPPER FPADD_conv_adapter(.X(adapter_z[m]), .Y(fp_fmap[m]), .R(convadapter_z[m])); 

		end
	endgenerate

	generate 
		for (m = 0; m < SA_SIZE; m++) begin: DELTA_ERROR

 
			relu_deriv relu_deriv(.enable(enable_convadapt_d), .clk(clk), .rst(rst), .in(convadapter_z[m][15]), .out(acc_deriv_SA[m]));
			FPMult_WRAPPER FPMULT_delta_error(.X(acc_deriv_SA[m]), .Y(error_IN[m]), .R(error_OUT_SA[m]));

		end
	endgenerate

	adder bias_adder(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_db), .acc(error_OUT_SA[j]), .sum(bpBchange_SA[0]));

    generate 
		for (m = 0; m < SA_SIZE; m++) begin: DELTA_WEIGHTS

            FPMult_WRAPPER FPMULT_delta_weight(.X(error_OUT_SA[m]), .Y(fp_fmap[m]), .R(actv_in_x_error_out0[m]));

		end
	endgenerate

    adder weight_adder0(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw0), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[0]));
    adder weight_adder1(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw1), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[1]));    
    adder weight_adder2(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw2), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[2]));
    adder weight_adder3(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw3), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[3]));
    adder weight_adder4(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw4), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[4]));
    adder weight_adder5(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw5), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[5]));


endmodule
