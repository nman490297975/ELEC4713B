`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
//
// Engineer: Nicholas Manning and Niyana Yohannes
// 
// Design Name: CNN_PA
// Module Name: parallel_adapter
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

// 32 Bit IEEE Floating Point Rep.
`define BIT_WIDTH 16                    //0.20
   

module PA_topmodule #(IMAGE_SIZE = 28*28, IMAGE_WIDTH = 28, IMAGE_CHANNELS = 1, FMAP_SIZE_PA = 28*28, FMAP_WIDTH_PA = 28, FMAP_CHANNELS = 6, PA_SIZE = 28*28, PA_KERNELS = 1, BW = `BIT_WIDTH+2-1) (
  
    /*------------------------------------------------------------------------------
    --  2. I/O Control Signals
    ------------------------------------------------------------------------------*/

    input logic                                         clk, rst, do_fp, 
    output logic                                        done_FP, done_BP,

    /*------------------------------------------------------------------------------
    --  3. I/O Data Signals
    ------------------------------------------------------------------------------*/

    input logic [IMAGE_CHANNELS*IMAGE_SIZE-1:0]   [BW:0]    image_IN,      
    input logic [PA_KERNELS*FMAP_SIZE_PA-1:0]     [BW:0]    fp_fmap,       

    input logic [PA_KERNELS-1:0]                  [BW:0]    weights_PA,   
    input logic [PA_KERNELS-1:0]                  [BW:0]    biases_PA,     

    input logic [PA_KERNELS*FMAP_SIZE_PA-1:0]     [BW:0]    error_IN,      

    output logic [PA_KERNELS*PA_SIZE-1:0]         [BW:0]    adapter_z,     
    output logic [PA_KERNELS*PA_SIZE-1:0]         [BW:0]    acc_deriv_PA,   
    output logic [PA_KERNELS*PA_SIZE-1:0]         [BW:0]    error_OUT_PA,  
    output logic [PA_KERNELS-1:0]                 [BW:0]    bpWchange_PA,  
    output logic [PA_KERNELS-1:0]                 [BW:0]    bpBchange_PA   

);

    /*------------------------------------------------------------------------------
    --  4. Control Signals and Constants
    ------------------------------------------------------------------------------*/

    logic                                   clear;

    logic [PA_SIZE-1:0] [BW:0]              adapter_z0; 
	logic [PA_SIZE-1:0] [BW:0]              convadapter_z; 
	
	logic [PA_SIZE-1:0] [BW:0]              actv_in_x_error_out;

    logic                                   enable_conv0, enable_convadapt_d, enable_accum_bp;
	integer                                 j;

	enum logic [2:0] {S_FPIDLE, S_FPROP_ADAPTER, S_FPROP_ADAPTER_BIAS, S_FPROP_CONVADAPTER_SUM, S_FPROP_CONVADAPTER_ACTV, S_BPROP_CONVADAPTER_DERIV, S_BPROP_ACCUM, S_DONE} fpcs, fpns;

    /*------------------------------------------------------------------------------
    --  5. FSM State Logic
    ------------------------------------------------------------------------------*/

	always_ff @(posedge clk, posedge rst)
		if (rst) fpcs <= S_FPIDLE;
		else fpcs <= fpns;
	

	always_comb begin
		done_FP = 1'b0;
        done_BP = 1'b0;
		clear = 1'b0;
		enable_conv0 = 1'b0;
		enable_convadapt_d = 1'b0;
		enable_accum_bp = 1'b0;

		case (fpcs)

			S_FPIDLE: begin
				clear = do_fp;
				fpns = (do_fp) ? S_FPROP_ADAPTER : S_FPIDLE;
			end

			S_FPROP_ADAPTER: begin
				enable_conv0 = 1'b1;
				fpns = S_FPROP_ADAPTER_BIAS;
			end

			S_FPROP_ADAPTER_BIAS: begin
				fpns = S_FPROP_CONVADAPTER_SUM; 
			end

			S_FPROP_CONVADAPTER_SUM: begin
				fpns = S_FPROP_CONVADAPTER_ACTV;
			end

            S_FPROP_CONVADAPTER_ACTV: begin
				fpns = S_BPROP_CONVADAPTER_DERIV; 
			end

			S_BPROP_CONVADAPTER_DERIV: begin
                done_FP = 1'b1;
				enable_convadapt_d = 1'b1;
				fpns = S_BPROP_ACCUM; 
			end

			S_BPROP_ACCUM: begin
                enable_accum_bp = 1'b1;
				fpns = (j == PA_SIZE - 1) ? S_DONE : S_BPROP_ACCUM;
			end

			S_DONE: begin
				done_BP = 1'b1;
				fpns = S_FPIDLE;
			end

			default: fpns = S_FPIDLE;

			endcase
	end

    /*------------------------------------------------------------------------------
    --  6. FSM Indexer
    ------------------------------------------------------------------------------*/

	always_ff @(posedge clk, posedge rst) begin

		if (rst) begin
            j <= '0;
		end
		
		else begin
			case (fpcs)

				S_FPIDLE: begin
                    j <= '0;
				end

                //Loop through channel
				S_BPROP_ACCUM: begin
					j <= (j < PA_SIZE - 1) ? (j + 1) : 0;
				end

				default: ;

			endcase
		end
	end


    /*------------------------------------------------------------------------------
    --  6. Forward Propgagtion Generation
    ------------------------------------------------------------------------------*/

	genvar m;
	generate
		for (m = 0; m < PA_SIZE; m++) begin: ADAPTER

            neuron neuron_conv(.clk(clk), .rst(rst), .clear(clear), .en(enable_conv0), .weight(weights_PA[0]), .data(image_IN[m]), .accum(adapter_z0[m]));
            FPADD_WRAPPER FPADD_bias(.X(adapter_z0[m]), .Y(biases_PA[0]), .R(adapter_z[m]));

		end
	endgenerate

	generate
		for (m = 0; m < PA_SIZE; m++) begin: CONV_ADAPTER

            FPADD_WRAPPER FPADD_convadapter(.X(adapter_z[m]), .Y(fp_fmap[m]), .R(convadapter_z[m]));

		end
	endgenerate
 
    /*------------------------------------------------------------------------------
    --  6. Back Propgagtion Generation
    ------------------------------------------------------------------------------*/

	generate
		for (m = 0; m < PA_SIZE; m++) begin: DELTA_ERROR

			relu_deriv relu_deriv(.enable(enable_convadapt_d), .clk(clk), .rst(rst), .in(convadapter_z[m][15]), .out(acc_deriv_PA[m]));
			FPMult_WRAPPER FPMULT_error(.X(acc_deriv_PA[m]), .Y(error_IN[m]), .R(error_OUT_PA[m]));
			
		end
	endgenerate

	adder bias_adder(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp), .acc(error_OUT_PA[j]), .sum(bpBchange_PA));

	generate
		for (m = 0; m < PA_SIZE; m++) begin: DELTA_WEIGHTS

		FPMult_WRAPPER FPMULT_delta_weights(.X(error_OUT_PA[m]), .Y(image_IN[m]), .R(actv_in_x_error_out[m]));

		end
	endgenerate
    
    adder weight_adder(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp), .acc(actv_in_x_error_out[j]), .sum(bpWchange_PA));

endmodule