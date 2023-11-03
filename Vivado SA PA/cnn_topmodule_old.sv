`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Nicholas Manning and Niyana Yohannes
// 
// Create Date: 
// Design Name: CNN_PA
// Module Name: net
// Project Name: Convolutional Adapter
// Target Devices: 
// Tool Versions: 
// Description: Proof of concept for a hardware convolutional layer serial adapter using a small CNN
// 
// Dependencies: 
// 
// Revision:
// Revision 0.02
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

/*------------------------------------------------------------------------------
--  
--  Structure
--
--  Admin
--  a. To Do
--  b. Design Notes
--
--  1. Define Constants
--  
--  External (I/O)
--  2. Control Signals
--  3. Data Signals
--  
--  Internal
--  4. Control Signals / Constants
--  5. FP RAM
--  6. BP RAM
--  7. BP Calculation
--
--  Hardware Generation
--  8. Generate FP Units
--  9. Generate BP Units
--
--  FSM
--  10. FSM Logic
--  11. FSM Indexer
--  12. Reusable Offcuts
--  
------------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
--  a. To Do:
--
--  X each on when finished.
--
--  RTL network 
--    [] load weights/biases in from files (main conv weights/biases)
--    [] load random zero mean weights/biases for Adapter (from testbench or from file)
--    [] load feature map (FMAP) input for serial adapter (ADAPT == 1)
--    [] Check control signals, RAM, BP calculation, generate FP, generate BP, FSM
--
--  Layers
--    [] SA (conv_neuron.sv + conv_adder.sv + conv_activation.sv)
--    
--  b. Design Notes:
--  
--  * Variables names: <backprop><name>_<LAYER_CODE>_<temp/inter>  eg. biases_OUT or bpWChange_OUT_inter
--  * Input type set to logic.  Consistent with tb for signal detection, should not affect synthesis.
--
--  Adapter 
--
--     Component         Dimensions                               Comment
--  * Control:          Separate control signals for SA and PA   (for debugging it's super obvious which adapter has been generated)
--  * Weights RAM:      KERNELS * IN_CHANNELS * 1 * 1            ([6][6] = 36 for SA, [6] = 6 for PA)
--  * Accum RAM:        KERNELS * WIDTH * HEIGHT                 (written to FMAP_CHANNELS number of times)
--  * Activation RAM:   KERNELS * WIDTH * HEIGHT     
--  
------------------------------------------------------------------------------*/ 
      
/*------------------------------------------------------------------------------
--  1. Define Constants
------------------------------------------------------------------------------*/

// 32 Bit IEEE Floating Point Rep.
`define BIT_WIDTH 8                    //0.20


module cnn_topmodule #(FMAP_SIZE_SA = 24*24, FMAP_WIDTH_SA = 24, FMAP_CHANNELS = 6, SA_SIZE = 24*24, SA_KERNELS = 1, BW = `BIT_WIDTH+2-1) (
  
    /*------------------------------------------------------------------------------
    --  2. I/O Control Signals
    ------------------------------------------------------------------------------*/

    input logic                                         clk, rst, do_fp, 
    //do_bp,
    output logic                                        done_FP, done_BP,

    /*------------------------------------------------------------------------------
    --  3. I/O Data Signals
    ------------------------------------------------------------------------------*/

    input logic [SA_KERNELS*FMAP_SIZE_SA-1:0]  		[BW:0]    fp_fmap,               // 6*24*24  = 3456
    //input logic [SA_KERNELS*FMAP_SIZE_SA-1:0]     [BW:0]    fp_fmap_current,      // 1*24*24  = 3456


    input logic [SA_KERNELS*FMAP_CHANNELS-1:0]    [BW:0]    weights_SA,   // Kernels * FMAP channels = 1*6 = 6
    input logic [SA_KERNELS-1:0]                  [BW:0]    biases_SA,    // 1 per kernel = 1 

    input logic [SA_KERNELS*FMAP_SIZE_SA-1:0]      [BW:0]    error_IN,     // 1*24*24 (error from actv(conv+adapt) layer)

    output logic [SA_KERNELS*SA_SIZE-1:0]         [BW:0]    adapter_z,    // 1*24*24  = 576

    output logic [SA_KERNELS*SA_SIZE-1:0]         [BW:0]    acc_deriv_SA, // 1*24*24  = 576
    output logic [SA_KERNELS*SA_SIZE-1:0]         [BW:0]    error_OUT_SA, // 1*24*24  = 576 (error propogated backwards from adapter layer)
    output logic [SA_KERNELS*FMAP_CHANNELS-1:0]   [BW:0]    bpWchange_SA, // Kernels * FMAP channels = 1*6 = 6
    output logic [SA_KERNELS-1:0]                 [BW:0]    bpBchange_SA // 1 per kernel = 1

);

    /*------------------------------------------------------------------------------
    --  4. Control Signals and Constants
    ------------------------------------------------------------------------------*/

    logic                                   clear;

    logic [SA_SIZE-1:0] [BW:0]              adapter_z0; // Conv data: eg. neuron.accum(s00_temp) --> adder.sum(s00) --> sigmoid.out(y0)
	logic [SA_SIZE-1:0] [BW:0]              convadapter_z; // Conv data: eg. neuron.accum(s00_temp) --> adder.sum(s00) --> sigmoid.out(y0)
	logic [SA_SIZE-1:0] [BW:0]              convadapter_actv; // Conv data: eg. neuron.accum(s00_temp) --> adder.sum(s00) --> sigmoid.out(y0)

	logic [SA_SIZE-1:0] [BW:0]              actv_in_x_error_out0, actv_in_x_error_out1, actv_in_x_error_out2, actv_in_x_error_out3, actv_in_x_error_out4, actv_in_x_error_out5; // Conv data: eg. neuron.accum(s00_temp) --> adder.sum(s00) --> sigmoid.out(y0)

    logic                                   enable_conv0, enable_convadapt_s, enable_convadapt_d, enable_accum_bp_db;
	logic 									enable_accum_bp_dw0, enable_accum_bp_dw1, enable_accum_bp_dw2, enable_accum_bp_dw3, enable_accum_bp_dw4, enable_accum_bp_dw5;
	integer                                 k, j; 
	//l;

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

            //To add bias to adapter z
			S_FPROP_CONV0_b: begin
				fpns = S_FPROP_CONV0_s;
			end
    
			//Add adapter with feature map;
			S_FPROP_CONV0_s: begin
				fpns = S_FPROP_CONVADAPT_s;
			end

			//Add activation function to conv+adapt
            S_FPROP_CONVADAPT_s: begin
				done_FP = 1'b1;
				enable_convadapt_s = 1'b1;
				fpns = S_BPROP_CONVADAPT_d;
			end

			//Get the derivative of relu
			S_BPROP_CONVADAPT_d: begin
				enable_convadapt_d = 1'b1;
				fpns = S_BPROP_ACCUM_db;
			end

			//Might need an extra cycle for deriv to propogate to mult//

			//Loop through channel
			S_BPROP_ACCUM_db: begin
				enable_accum_bp_db = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw0 : S_BPROP_ACCUM_db;
			end

			//Loop through channel
			S_BPROP_ACCUM_dw0: begin
				enable_accum_bp_dw0 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw1 : S_BPROP_ACCUM_dw0;
			end
			//Loop through channel
			S_BPROP_ACCUM_dw1: begin
				enable_accum_bp_dw1 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw2 : S_BPROP_ACCUM_dw1;
			end
			//Loop through channel
			S_BPROP_ACCUM_dw2: begin
				enable_accum_bp_dw2 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw3 : S_BPROP_ACCUM_dw2;
			end
			//Loop through channel
			S_BPROP_ACCUM_dw3: begin
				enable_accum_bp_dw3 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw4 : S_BPROP_ACCUM_dw3;
			end
			//Loop through channel
			S_BPROP_ACCUM_dw4: begin
				enable_accum_bp_dw4 = 1'b1;
				fpns = (j == SA_SIZE - 1) ? S_BPROP_ACCUM_dw5 : S_BPROP_ACCUM_dw4;
			end
			//Loop through channel
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
			// l <= '0;
		end
		
		else begin
			case (fpcs)

				S_FPIDLE: begin
					k <= '0;
					j <= '0;
					// l <= '0;
				end

				/// Adapter for loop
				/// Groups of 6 neurons = 6 times
				S_FPROP_CONV0: begin
					k <= (k < FMAP_CHANNELS - 1) ? (k + 1) : 0;
				end

				//Loop through channel
				S_BPROP_ACCUM_db: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end

				//Loop through channel
				S_BPROP_ACCUM_dw0: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				//Loop through channel
				S_BPROP_ACCUM_dw1: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				//Loop through channel
				S_BPROP_ACCUM_dw2: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				//Loop through channel
				S_BPROP_ACCUM_dw3: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				//Loop through channel
				S_BPROP_ACCUM_dw4: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end
				//Loop through channel
				S_BPROP_ACCUM_dw5: begin
					j <= (j < SA_SIZE - 1) ? (j + 1) : 0;
				end

				default: ;

			endcase
		end
	end



	///
	/// Adapter generator
	///
	genvar m;
	generate //Adapter layer
		for (m = 0; m < SA_SIZE; m++) begin: CONV_R

            /// 6*m as 1 feature map every 6 numbers 
            // +k as we want to accumulate 6 numbers 

            neuron neuron_conv (.clk(clk), .rst(rst), .clear(clear), .en(enable_conv0), .weight(weights_SA[k]), .data(fp_fmap[m]), .accum(adapter_z0[m]));
            FPADD_11bit_WRAPPER FPADD_bias(.X(adapter_z0[m]), .Y(biases_SA[0]), .R(adapter_z[m])); 

		end
	endgenerate

	///
	/// Adapter+Convolution generator
	///
	generate //Adapter+Convolution layer
		for (m = 0; m < SA_SIZE; m++) begin: CONVADAPT

            FPADD_11bit_WRAPPER FPADD_convadapt(.X(adapter_z[m]), .Y(fp_fmap[m]), .R(convadapter_z[m])); 

		end
	endgenerate

	// /
	// / Backprop generator
	// /
	generate //Adapter+Convolution layer
		for (m = 0; m < SA_SIZE; m++) begin: error

            //NOTE//
            //How maxpool is read in affects how to calculate bp_SA_delta_error//
 
			relu_deriv relu_deriv(.enable(enable_convadapt_d), .clk(clk), .rst(rst), .in(convadapter_z[m][7]), .out(acc_deriv_SA[m]));
			FPMult_11bit_WRAPPER FPMULT_error(.X(acc_deriv_SA[m]), .Y(error_IN[m]), .R(error_OUT_SA[m]));
			//.C(32'b0), .negateAB(1'b0), .negateC(1'b0), .RndMode(2'b00), 
		end
	endgenerate

	
	//To get change in weights and bias, need to loop over a whole channel as every neuron in a channel contributes to the change. Thus will need 576 cycles to add 24*24 = 576 numbers.
    //Increment j by 1 until 24*24 and adder is adding the whole time
	// Set enable_accum_bp to 1 after finished to get final adder value

	///
	/// Change in bias generator
	///
	adder bias_adder(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_db), .acc(error_OUT_SA[j]), .sum(bpBchange_SA[0]));

	///
	/// Change in weight generator
	///
    generate //actv_in_x_error_out generate
		for (m = 0; m < SA_SIZE; m++) begin: A_in_X_error_out

            FPMult_11bit_WRAPPER FPMULT_ain_x_eout0(.X(error_OUT_SA[m]), .Y(fp_fmap[m]), .R(actv_in_x_error_out0[m])); //Can change to just 1 channe;
            //.C(32'b0), .negateAB(1'b0), .negateC(1'b0), .RndMode(2'b00), 
		end
	endgenerate

    adder weight_adder0(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw0), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[0]));
    adder weight_adder1(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw1), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[1]));    
    adder weight_adder2(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw2), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[2]));
    adder weight_adder3(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw3), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[3]));
    adder weight_adder4(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw4), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[4]));
    adder weight_adder5(.clk(clk), .rst(rst), .clear(clear), .enable(enable_accum_bp_dw5), .acc(actv_in_x_error_out0[j]), .sum(bpWchange_SA[5]));


endmodule
